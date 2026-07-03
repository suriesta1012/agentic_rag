# vectorstore.py
"""
ChromaDB-backed vector store.

ChromaDB runs in-process and persists data to a local folder —
no Docker, no JVM, no external service required.
"""

from functools import lru_cache

from langchain_community.vectorstores import Chroma

from config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR
from core.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Return a cached Chroma vector store instance.
    The store is created (or re-opened) lazily on first use.
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
