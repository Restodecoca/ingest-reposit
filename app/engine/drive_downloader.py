try:
    import os
    import io
    import json
    import hashlib
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from tqdm import tqdm
except ImportError as e:
    # Se faltarem as bibliotecas necessárias, levantamos Exception
    raise Exception(
        "Faltam bibliotecas necessárias para o GoogleDriveDownloader. "
        "Instale-as com:\n\n"
        "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib tqdm\n\n"
        f"Detalhes do erro: {str(e)}"
    )

class GoogleDriveDownloader:
    """
    Classe para autenticar e baixar arquivos do Google Drive,
    preservando a estrutura de pastas e evitando downloads redundantes.
    - Nunca abrirá navegador se não encontrar token válido (apenas levanta exceção).
    - Pode ler 'credentials.json' e 'token.json' do disco ou das variáveis de ambiente.
    """

    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self, chunksize=100 * 1024 * 1024):
        """
        :param chunksize: Tamanho (em bytes) de cada chunk ao baixar arquivos.
                          Ex.: 100MB = 100 * 1024 * 1024.
        """
        self.chunksize = chunksize
        self.service = None

    def _get_credentials_from_env_or_file(self):
        """
        Verifica se existem variáveis de ambiente para 'CREDENTIALS' e 'TOKEN'.
        Caso contrário, tenta usar arquivos locais 'credentials.json' e 'token.json'.
        
        Se o token local/ambiente não existir ou for inválido (sem refresh),
        levanta exceção (não abrimos navegador neste fluxo).
        """
        print("Procurando credentials na variavel de ambiente...")
        env_credentials = os.environ.get("CREDENTIALS")  # Conteúdo JSON do client secrets
        env_token = os.environ.get("TOKEN")              # Conteúdo JSON do token

        creds = None

        # 1) Carregar credenciais do ambiente, se houver
        if env_credentials:
            try:
                creds_json = json.loads(env_credentials)
            except json.JSONDecodeError:
                raise ValueError("A variável de ambiente 'CREDENTIALS' não contém JSON válido.")
            
            # Validamos o "client_id" para garantir que seja um JSON de credenciais mesmo
            client_id = (
                creds_json.get("installed", {}).get("client_id") or
                creds_json.get("web", {}).get("client_id")
            )
            if not client_id:
                raise ValueError("Credenciais em memória não parecem válidas. Faltam campos 'client_id'.")

        else:
            # Se não há credenciais no ambiente, tentamos local
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "Nenhuma credencial encontrada em ambiente ou no arquivo 'credentials.json'."
                )
            print("Variavel não encontrada, usando credentials.json")
            with open("credentials.json", 'r', encoding='utf-8') as f:
                creds_json = json.load(f)

        print("Procurando tokens na variavel de ambiente...")
        token_data = None
        if env_token:
            try:
                token_data = json.loads(env_token)
            except json.JSONDecodeError:
                raise ValueError("A variável de ambiente 'TOKEN' não contém JSON válido.")
        else:
            # Se não há token no ambiente, checamos arquivo local
            if os.path.exists("token.json"):
                print("Variavel não encontrada, usando token.json")
                with open("token.json", 'r', encoding='utf-8') as tf:
                    token_data = json.load(tf)
            else:
                raise FileNotFoundError(
                    "Não há token no ambiente nem em 'token.json'. "
                    "Não é possível autenticar sem abrir navegador, então abortando."
                )
        
        # 3) Criar credenciais a partir do token_data
        creds = Credentials.from_authorized_user_info(token_data, self.SCOPES)

        # 4) Se expirou, tenta refresh
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Salva token atualizado, se estiver usando arquivo local
                if not env_token:  # só sobrescreve se está lendo do disco
                    with open("token.json", 'w', encoding='utf-8') as token_file:
                        token_file.write(creds.to_json())
            else:
                # Se não é válido e não há refresh token, não temos como renovar sem navegador
                raise RuntimeError(
                    "As credenciais de token são inválidas/expiradas e sem refresh token. "
                    "Não é possível abrir navegador neste fluxo, abortando."
                )

        return creds

    def authenticate(self):
        """Cria e armazena o serviço do Drive API nesta instância."""
        creds = self._get_credentials_from_env_or_file()
        self.service = build("drive", "v3", credentials=creds)

    def _list_files_in_folder(self, folder_id):
        """Retorna a lista de itens (arquivos/pastas) diretamente em 'folder_id'."""
        items = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed=false"

        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            items.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if not page_token:
                break
        return items

    def _get_file_metadata(self, file_id):
        """
        Retorna (size, md5Checksum, modifiedTime) de um arquivo no Drive.
        Se algum campo não existir, retorna valor padrão.
        """
        data = self.service.files().get(
            fileId=file_id,
            fields='size, md5Checksum, modifiedTime'
        ).execute()

        size = int(data.get('size', 0))
        md5 = data.get('md5Checksum', '')
        modified_time = data.get('modifiedTime', '')
        return size, md5, modified_time

    def _get_all_items_recursively(self, folder_id, parent_path=''):
        """
        Percorre recursivamente a pasta (folder_id) no Drive,
        retornando lista de dicts (id, name, mimeType, path).
        """
        results = []
        items = self._list_files_in_folder(folder_id)

        for item in items:
            current_path = os.path.join(parent_path, item['name'])
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                results.append({
                    'id': item['id'],
                    'name': item['name'],
                    'mimeType': item['mimeType'],
                    'path': current_path
                })
                sub = self._get_all_items_recursively(item['id'], current_path)
                results.extend(sub)
            else:
                results.append({
                    'id': item['id'],
                    'name': item['name'],
                    'mimeType': item['mimeType'],
                    'path': parent_path
                })
        return results

    def _needs_download(self, local_folder, file_info):
        """
        Verifica se o arquivo em 'file_info' precisa ser baixado.
        - Se não existir localmente, retorna True.
        - Se existir, compara tamanho e MD5 (quando disponível).
        - Retorna True se for diferente, False se for idêntico.
        """
        file_id = file_info['id']
        file_name = file_info['name']
        rel_path = file_info['path']

        drive_size, drive_md5, _ = self._get_file_metadata(file_id)
        full_local_path = os.path.join(local_folder, rel_path, file_name)

        if not os.path.exists(full_local_path):
            return True  # Não existe localmente

        local_size = os.path.getsize(full_local_path)
        if local_size != drive_size:
            return True

        if drive_md5:
            with open(full_local_path, 'rb') as f:
                local_md5 = hashlib.md5(f.read()).hexdigest()
            if local_md5 != drive_md5:
                return True

        return False

    def _download_single_file(self, file_id, file_name, relative_path, progress_bar):
        """
        Faz download de um único arquivo do Drive, atualizando a barra de progresso global.
        """
        # Como fizemos 'os.chdir(local_folder)' antes, 'relative_path' pode ser vazio.
        # Então concatenamos sem o local_folder:
        file_path = os.path.join(relative_path, file_name)

        # Se o path do diretório for vazio, cai no '.' para evitar WinError 3
        dir_name = os.path.dirname(file_path) or '.'
        os.makedirs(dir_name, exist_ok=True)

        request = self.service.files().get_media(fileId=file_id)
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=self.chunksize)
            done = False
            previous_progress = 0

            while not done:
                status, done = downloader.next_chunk()
                if status:
                    current_progress = status.resumable_progress
                    chunk_downloaded = current_progress - previous_progress
                    previous_progress = current_progress
                    progress_bar.update(chunk_downloaded)

    def download_from_folder(self, drive_folder_id: str, local_folder: str):
        """
        Método principal para:
         1. Autenticar sem abrir navegador (usa token local/ambiente).
         2. Exibir "Iniciando verificação de documentos".
         3. Listar recursivamente arquivos da pasta do Drive.
         4. Verificar quais precisam de download.
         5. Baixar apenas o necessário, com barra de progresso única.
        """
        print("Iniciando verificação de documentos")

        if not self.service:
            self.authenticate()

        print("Buscando lista de arquivos no Drive...")
        all_items = self._get_all_items_recursively(drive_folder_id)

        # Filtra apenas arquivos (exclui subpastas)
        all_files = [f for f in all_items if f['mimeType'] != 'application/vnd.google-apps.folder']

        print("Verificando quais arquivos precisam ser baixados...")
        files_to_download = []
        total_size_to_download = 0
        for info in all_files:
            if self._needs_download(local_folder, info):
                drive_size, _, _ = self._get_file_metadata(info['id'])
                total_size_to_download += drive_size
                files_to_download.append(info)

        if not files_to_download:
            print("Nenhum arquivo novo ou atualizado. Tudo sincronizado!")
            return

        print("Calculando total de bytes a serem baixados...")

        # Ajusta a pasta local e cria se necessário
        os.makedirs(local_folder, exist_ok=True)

        # Muda diretório de trabalho para simplificar criação de subpastas
        old_cwd = os.getcwd()
        os.chdir(local_folder)

        # Cria a barra de progresso global
        progress_bar = tqdm(
            total=total_size_to_download,
            unit='B',
            unit_scale=True,
            desc='Baixando arquivos'
        )

        # Baixa só o que precisa
        for file_info in files_to_download:
            self._download_single_file(
                file_id=file_info['id'],
                file_name=file_info['name'],
                relative_path=file_info['path'],
                progress_bar=progress_bar
            )

        progress_bar.close()
        os.chdir(old_cwd)   

        print("Download concluído com sucesso!")
