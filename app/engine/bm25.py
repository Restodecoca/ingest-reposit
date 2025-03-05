from dotenv import load_dotenv

load_dotenv()

import logging
import os
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Definir o diretório de armazenamento
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
BM25_PATH = os.getenv("BM25_PATH", os.path.join(STORAGE_DIR, "bm25"))
top_k = int(os.getenv("TOP_K", 2))

def get_bm25_retriever():
    docstore = SimpleDocumentStore.from_persist_dir(STORAGE_DIR)

    documents = []
    for doc_id, doc_data in docstore.docs.items():
        document = Document(
            text=doc_data.text,
            id_=doc_data.id_,
            metadata=doc_data.metadata if doc_data.metadata else {}  
        )
        document.metadata["private"] = "false"
        documents.append(document)

    splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)


    if nodes:
        bm25_dir = BM25_PATH
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
            language="portuguese",
            verbose=True
        )
        os.makedirs(bm25_dir, exist_ok=True)
        bm25_retriever.persist(bm25_dir)
        logger.info("BM25Retriever persistido em %s", bm25_dir)
    else:
        logger.warning("Nenhum nó gerado. Pulando criação do BM25Retriever.")

