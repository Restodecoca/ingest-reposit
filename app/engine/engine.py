import os

from app.engine.index import IndexConfig, get_index
from app.engine.node_postprocessors import NodeCitationProcessor
from fastapi import HTTPException
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from app.engine.mysqlchatstore import MySQLChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")



# Configuração do Chat Store no MySQL
chat_store = MySQLChatStore.from_params(
    host=os.getenv("MYSQL_HOST"), 
    port=int(os.getenv("MYSQL_PORT", 3306)), 
    user=os.getenv("MYSQL_USER"), 
    password=os.getenv("MYSQL_PASSWORD"), 
    database=os.getenv("MYSQL_DATABASE"),
    table_name=os.getenv("MYSQL_TABLE", "chatstore")
)

def get_chat_engine(params=None, event_handlers=None, **kwargs):
    system_prompt = os.getenv("SYSTEM_PROMPT")
    citation_prompt = os.getenv("SYSTEM_CITATION_PROMPT", None)
    context_prompt = os.getenv("SYSTEM_CONTEXT_PROMPT", None)
    top_k = int(os.getenv("TOP_K", 2))
    llm = Settings.llm
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=llm.metadata.context_window - 256,
        chat_store=chat_store,
        chat_store_key=kwargs.get("session_id", "Sicoob") #ESPERANDO LOGIN
    )
    callback_manager = CallbackManager(handlers=event_handlers or [])

    node_postprocessors = []
    if citation_prompt:
        node_postprocessors = [NodeCitationProcessor()]
        system_prompt = f"{system_prompt}\n{citation_prompt}"

    index_config = IndexConfig(callback_manager=callback_manager, **(params or {}))
    index = get_index(index_config)
    if index is None:
        raise HTTPException(
            status_code=500,
            detail=str(
                "StorageContext is empty - call 'poetry run generate' to generate the storage first"
            ),
        )
    if top_k != 0 and kwargs.get("similarity_top_k") is None:
        kwargs["similarity_top_k"] = top_k
    index_retriever = index.as_retriever(**kwargs)
    bm25_dir = os.getenv("BM25_PATH")
    if os.path.exists(bm25_dir):
        bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
        bm25_retriever.similarity_top_k = top_k
        bm25_retriever.language = "portuguese"  
    else:
        raise HTTPException(
            status_code=500,
            detail=f"BM25Retriever is empty - call 'poetry run generate' to generate the storage first"
        )

    retriever = QueryFusionRetriever(
        [index_retriever, bm25_retriever],
        similarity_top_k=top_k,
        mode="reciprocal_rerank",
        num_queries=1,
        use_async=True,
        verbose=True,
        callback_manager=callback_manager,
    )

    return CondensePlusContextChatEngine(
        llm=llm,
        memory=memory,
        system_prompt=system_prompt,
        context_prompt=context_prompt,
        retriever=retriever,
        node_postprocessors=node_postprocessors,  # type: ignore
        callback_manager=callback_manager,
    )
