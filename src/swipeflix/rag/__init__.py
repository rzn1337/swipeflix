"""RAG (Retrieval-Augmented Generation) module for SwipeFlix."""

from swipeflix.rag.generator import RAGGenerator
from swipeflix.rag.ingest import DocumentIngester, MovieDocument
from swipeflix.rag.retriever import MovieRetriever

__all__ = [
    "DocumentIngester",
    "MovieDocument",
    "MovieRetriever",
    "RAGGenerator",
]
