# flake8: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import logging
import os

from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

from app.engine.loaders import get_documents
from app.engine.vectordb import get_vector_store
from app.engine.bm25 import get_bm25_retriever
from app.settings import init_settings
from app.engine.drive_downloader import GoogleDriveDownloader
from app.engine.document_creator import create_single_document_with_filenames

from app.config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
DRIVE_FOLDER = os.getenv("DRIVE_FOLDER")

def get_doc_store():
    # If the storage directory is there, load the document store from it.
    # If not, set up an in-memory document store since we can't load from a directory that doesn't exist.
    if os.path.exists(STORAGE_DIR):
        return SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    else:
        return SimpleDocumentStore()


def run_pipeline(docstore, vector_store, documents):
    
    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(chunk_size=Settings.chunk_size,chunk_overlap=Settings.chunk_overlap,),Settings.embed_model,],
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,  # type: ignore
        vector_store=vector_store,
    )

    # Run the ingestion pipeline and store the results
    nodes = pipeline.run(show_progress=True, documents=documents)

    return nodes


def persist_storage(docstore, vector_store):
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )
    storage_context.persist(STORAGE_DIR)


def generate_datasource():
    init_settings()
    GoogleDriveDownloader().download_from_folder(DRIVE_FOLDER, DATA_DIR)
    logger.info("Generate index for the provided data")

    # Get the stores and documents or create new ones
    documents = get_documents()
    document = create_single_document_with_filenames(DATA_DIR)
    documents.append(document)
    for doc in documents:
        doc.metadata["private"] = "false"
    docstore = get_doc_store()
    vector_store = get_vector_store()

    # Run the ingestion pipeline
    _ = run_pipeline(docstore, vector_store, documents)

    persist_storage(docstore, vector_store)

    get_bm25_retriever()

    logger.info("Finished generating the index")


if __name__ == "__main__":
    generate_datasource()
