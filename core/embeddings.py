# embeddings.py
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from config import EMBEDDING_MODEL

def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)