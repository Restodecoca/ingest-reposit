from typing import Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic import Field
import pymysql

from llama_index.core.storage.chat_store import BaseChatStore
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer



class MySQLChatStore(BaseChatStore):
    """
    Implementação de um ChatStore que armazena mensagens em uma tabela MySQL,
    unindo a pergunta do usuário e a resposta do assistente na mesma linha.
    """
    table_name: Optional[str] = Field(default="chatstore", description="Nome da tabela MySQL.")

    _session: Optional[sessionmaker] = None
    _async_session: Optional[sessionmaker] = None

    def __init__(self, session: sessionmaker, async_session: sessionmaker, table_name: str):
        super().__init__(table_name=table_name.lower())
        self._session = session
        self._async_session = async_session
        self._initialize()

    @classmethod
    def from_params(cls, host: str, port: str, database: str, user: str, password: str, table_name: str = "chatstore") -> "MySQLChatStore":
        """
        Cria o sessionmaker síncrono e assíncrono, retornando a instância da classe.
        """
        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        async_conn_str = f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"
        session, async_session = cls._connect(conn_str, async_conn_str)
        return cls(session=session, async_session=async_session, table_name=table_name)

    @classmethod
    def _connect(cls, connection_string: str, async_connection_string: str) -> tuple[sessionmaker, sessionmaker]:
        """
        Cria e retorna um sessionmaker síncrono e um sessionmaker assíncrono.
        """
        engine = create_engine(connection_string, echo=False, pool_pre_ping=True, pool_recycle=3600)
        session = sessionmaker(bind=engine)

        async_engine = create_async_engine(async_connection_string)
        async_session = sessionmaker(bind=async_engine, class_=AsyncSession)

        return session, async_session

    def _initialize(self):
        """
        Garante que a tabela exista, com colunas para armazenar user_input e response.
        """
        with self._session() as session:
            session.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chat_store_key VARCHAR(255) NOT NULL,
                    user_input TEXT,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            session.commit()

    def get_keys(self) -> list[str]:
        """
        Retorna todas as chaves armazenadas.
        """
        with self._session() as session:
            result = session.execute(text(f"""
                SELECT DISTINCT chat_store_key FROM {self.table_name}
            """))
            return [row[0] for row in result.fetchall()]

    def get_messages(self, key: str) -> list[ChatMessage]:
        """
        Retorna a conversa inteira (perguntas e respostas), na ordem de inserção (id).
        Cada linha pode conter o user_input, o response ou ambos (caso já respondido).
        """
        with self._session() as session:
            rows = session.execute(text(f"""
                SELECT user_input, response
                FROM {self.table_name}
                WHERE chat_store_key = :key
                ORDER BY id
            """), {"key": key}).fetchall()

            messages = []
            for user_in, resp in rows:
                if user_in is not None:
                    messages.append(ChatMessage(role='user', content=user_in))
                if resp is not None:
                    messages.append(ChatMessage(role='assistant', content=resp))
            return messages

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """
        Sobrescreve o histórico de mensagens de uma chave (apaga tudo e insere novamente).
        Se quiser somente acrescentar, use add_message.
        
        Aqui, cada pergunta do usuário gera uma nova linha.
        Assim que encontrar uma mensagem de assistente, atualiza essa mesma linha.
        Se houver assistentes sem usuários, insere normalmente.
        """
        with self._session() as session:
            # Limpa histórico anterior
            session.execute(text(f"""
                DELETE FROM {self.table_name} WHERE chat_store_key = :key
            """), {"key": key})

            # Reinsere na ordem
            current_id = None
            for msg in messages:
                if msg.role == 'user':
                    # Cria nova linha com user_input
                    result = session.execute(text(f"""
                        INSERT INTO {self.table_name} (chat_store_key, user_input)
                        VALUES (:key, :ui)
                    """), {"key": key, "ui": msg.content})
                    # Pega o id do insert
                    current_id = result.lastrowid

                else:
                    # Tenta atualizar a última linha se existir
                    if current_id is not None:
                        session.execute(text(f"""
                            UPDATE {self.table_name}
                            SET response = :resp
                            WHERE id = :id
                        """), {"resp": msg.content, "id": current_id})
                        # Depois de atualizar a linha, zera o current_id
                        current_id = None
                    else:
                        # Se não houver pergunta pendente, insere como nova linha
                        session.execute(text(f"""
                            INSERT INTO {self.table_name} (chat_store_key, response)
                            VALUES (:key, :resp)
                        """), {"key": key, "resp": msg.content})

            session.commit()

    def add_message(self, key: str, message: ChatMessage) -> None:
        """
        Acrescenta uma nova mensagem no fluxo. Se for do usuário, insere nova linha;
        se for do assistente, tenta preencher a linha pendente que não tenha resposta.
        """

        with self._session() as session:
            if message.role == 'user':
                # Sempre cria uma nova linha para mensagens de usuário
                insert_stmt = text(f"""
                    INSERT INTO {self.table_name} (chat_store_key, user_input)
                    VALUES (:key, :ui)
                """)
                session.execute(insert_stmt, {
                    "key": key,
                    "ui": message.content
                })
            else:
                # Tenta encontrar a última linha sem resposta
                
                row = session.execute(text(f"""
                    SELECT id
                    FROM {self.table_name}
                    WHERE chat_store_key = :key
                      AND user_input IS NOT NULL
                      AND response IS NULL
                    ORDER BY id DESC
                    LIMIT 1
               """), {"key": key}).fetchone()

                if row:
                    # Atualiza com a resposta
                    msg_id = row[0]
                    
                    update_stmt = text(f"""
                        UPDATE {self.table_name}
                        SET response = :resp
                        WHERE id = :id
                    """)
                    session.execute(update_stmt, {
                        "resp": message.content,
                        "id": msg_id
                    })
                else:
                    # Se não achar linha pendente, insere como nova
                   
                    insert_stmt = text(f"""
                        INSERT INTO {self.table_name} (chat_store_key, response)
                        VALUES (:key, :resp)
                    """)
                    session.execute(insert_stmt, {
                        "key": key,
                        "resp": message.content
                    })

            session.commit()
            


    def delete_messages(self, key: str) -> None:
        """
        Remove todas as linhas associadas a 'key'.
        """
        with self._session() as session:
            session.execute(text(f"""
                DELETE FROM {self.table_name} WHERE chat_store_key = :key
            """), {"key": key})
            session.commit()

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Apaga a última mensagem da conversa (considerando a ordem de inserção).
        Se a última linha tiver pergunta e resposta, remove primeiro a resposta;
        caso não exista resposta, remove a linha inteira.
        """
        with self._session() as session:
            # Localiza a última linha
            row = session.execute(text(f"""
                SELECT id, user_input, response
                FROM {self.table_name}
                WHERE chat_store_key = :key
                ORDER BY id DESC
                LIMIT 1
            """), {"key": key}).fetchone()

            if not row:
                return None

            row_id, user_in, resp = row

            # Se a linha tiver somente pergunta, apagamos a linha inteira.
            # Se tiver também a resposta, apagamos só a parte do assistente.
            if user_in and resp:
                # Remove a resposta
                session.execute(text(f"""
                    UPDATE {self.table_name}
                    SET response = NULL
                    WHERE id = :id
                """), {"id": row_id})
                session.commit()
                return ChatMessage(role='assistant', content=resp)
            else:
                # Deleta a linha inteira
                session.execute(text(f"""
                    DELETE FROM {self.table_name}
                    WHERE id = :id
                """), {"id": row_id})
                session.commit()

                if user_in:
                    return ChatMessage(role='user', content=user_in)
                elif resp:
                    return ChatMessage(role='assistant', content=resp)
                else:
                    return None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Deleta a mensagem com base na ordem total do histórico. O índice 'idx' é
        calculado após reconstruir a lista de ChatMessages (user e assistant).
        """
        messages = self.get_messages(key)
        if idx < 0 or idx >= len(messages):
            return None

        removed = messages[idx]

        # Agora precisamos traduzir 'idx' para saber qual registro no banco será modificado.
        # É mais simples recriar todos os dados com set_messages sem a mensagem em 'idx':
        messages.pop(idx)
        self.set_messages(key, messages)

        return removed
