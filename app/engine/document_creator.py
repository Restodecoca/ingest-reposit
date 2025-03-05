import os
import re
from pathlib import Path
from llama_index.core import Document

def create_single_document_with_filenames(directory_path: str) -> Document:
    """
    Iterates through the specified folder, organizes files into a structure by years (YYYY) and months (MM),
    generates a descriptive text similar to the previous code (without saving to a file) and returns
    this text encapsulated in a Document object.

    Usage:
        directory_path = "documentos"
        doc = create_single_document_with_filenames(directory_path)
        documents.append(doc)
    """
    # Dictionary {year: {month: [files]}}
    year_structure = {}
    # List for files that do not follow the YYYY-MM pattern
    undated_files = []
    # Set for all files (for an overall listing at the end)
    all_files = set()

    # Traverse the directory recursively
    for root, dirs, files in os.walk(directory_path):
        year_month_match = re.search(r"(\d{4})-(\d{2})", root)
        if year_month_match:
            year = year_month_match.group(1)
            month = year_month_match.group(2)

            if year not in year_structure:
                year_structure[year] = {}
            if month not in year_structure[year]:
                year_structure[year][month] = []

            for file_name in files:
                year_structure[year][month].append(file_name)
                all_files.add(file_name)
        else:
            # If the folder does not follow the YYYY-MM pattern, consider these files as "undated"
            for file_name in files:
                undated_files.append(file_name)
                all_files.add(file_name)

    # Build the final descriptive text using the defined cases
    final_description = []

    # Sort the list of years and iterate
    for year in sorted(year_structure.keys()):
        sorted_months = sorted(year_structure[year].keys())
        month_count = len(sorted_months)

        # Case 1: Only one month and one file
        if month_count == 1:
            unique_month = sorted_months[0]
            files_in_month = year_structure[year][unique_month]
            if len(files_in_month) == 1:
                full_month_name = month_full_name(unique_month)
                only_file = files_in_month[0]
                final_description.append(
                    f"No ano de {year}, temos somente o mês de {full_month_name} (mês {unique_month}), "
                    f"outros meses não foram listados, e nessa pasta encontramos apenas "
                    f"um manual, documento, arquivo, mps, mig ou organograma chamado {only_file}."
                )
                # Skip to the next year
                continue

        # Case 2: Multiple months or more than one file in a month
        initial_sentence = f"No ano de {year}, temos os meses de "
        months_description = [f"{month_full_name(m)} (mês {m})" for m in sorted_months]
        initial_sentence += ", ".join(months_description) + "."
        final_description.append(initial_sentence)

        for m in sorted_months:
            files_in_month = year_structure[year][m]
            full_month_name = month_full_name(m)
            files_count = len(files_in_month)
            if files_count == 1:
                final_description.append(
                    f"Em {full_month_name} temos somente o manual, documento, arquivo, mps, mig ou organograma chamado {files_in_month[0]}."
                )
            else:
                final_description.append(
                    f"Em {full_month_name} temos {files_count} manuais, documentos, arquivos, mps, migs ou organogramas, chamados: {', '.join(files_in_month)}."
                )

    # Case 3: Files without a date
    if undated_files:
        unique_undated_files = list(set(undated_files))
        if len(unique_undated_files) == 1:
            final_description.append(
                f"Em nossos documentos, fora de pastas de data, temos somente este arquivo: {unique_undated_files[0]}."
            )
        else:
            final_description.append(
                "Em nossos documentos, fora de pastas de data, encontramos estes arquivos: " + ", ".join(unique_undated_files) + "."
            )

    # Global sorted list of all files
    sorted_all_files = sorted(all_files)
    final_description.append(
        "Essas são as listas de todos os documentos, manuais, migs, mps e organogramas que podemos "
        f"resolver, listar e usar para nossas respostas, esses documentos vão ser muito úteis: {', '.join(sorted_all_files)}"
    )

    # Combine the final descriptive text
    document_text = "\n".join(final_description)

    # Create and return a Document object with this text
    document = Document(
        text=document_text,
        metadata={
            "description": "Lista de manuais, arquivos, documentos, organogramas e arquivos que podemos responder.",
            "file_name": "Lista de documentos",
            "keywords": "Manuais, Organogramas, Documentos, Arquivos, MPS, MIG",
            "summary": "Entre 2011 e 2024, há uma grande variedade de manuais e documentos organizados por ano e mês: alguns anos possuem apenas um mês e um único arquivo, enquanto outros registram diversos períodos, cada um com vários manuais. Adicionalmente, há alguns arquivos “soltos”, fora dessas pastas de data. No fim, existe uma listagem completa de todos os itens, cobrindo instruções bancárias, regulamentos internos e outros materiais de suporte."
        }
    )
    return document

def month_full_name(month_str: str) -> str:
    """
    Converts a month number string to its full name in Portuguese.
    For example, converts '07' to 'Julho' and '09' to 'Setembro'.
    """
    months_dict = {
        "01": "Janeiro",   "02": "Fevereiro", "03": "Março",
        "04": "Abril",     "05": "Maio",      "06": "Junho",
        "07": "Julho",     "08": "Agosto",    "09": "Setembro",
        "10": "Outubro",   "11": "Novembro",  "12": "Dezembro"
    }
    return months_dict.get(month_str, month_str)

