"""Retriever component for SwipeFlix RAG pipeline.

Handles semantic search over the movie document index using FAISS.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from swipeflix.monitoring.llm_metrics import (
    rag_documents_retrieved,
    rag_relevance_score,
    rag_retrieval_duration,
)
from swipeflix.rag.ingest import DocumentIngester, MovieDocument

try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except (ImportError, OSError, RuntimeError):
    SBERT_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MovieRetriever:
    """Semantic retriever for movie documents.

    Uses FAISS for efficient similarity search over document embeddings.
    Supports filtering, re-ranking, and hybrid search.
    """

    def __init__(
        self,
        index_path: str = "data/faiss_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k_default: int = 5,
        min_score_threshold: float = 0.1,  # Lower threshold for similarity scores (0-1 range)
    ):
        self.index_path = Path(index_path)
        self.top_k_default = top_k_default
        self.min_score_threshold = min_score_threshold

        # Initialize embedding model
        if SBERT_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(embedding_model)
                self.embedding_dim = (
                    self._embedding_model.get_sentence_embedding_dimension()
                )
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = None
                self.embedding_dim = 384
        else:
            self._embedding_model = None
            self.embedding_dim = 384

        # Load or initialize ingester
        self._ingester: Optional[DocumentIngester] = None
        self._load_index()

    def _load_index(self) -> bool:
        """Load the FAISS index from disk."""
        self._ingester = DocumentIngester(index_path=str(self.index_path))

        if self.index_path.exists():
            success = self._ingester.load_index()
            if success and len(self._ingester.documents) > 0:
                logger.info(
                    f"Loaded index with {len(self._ingester.documents)} documents"
                )
                return True
            elif success and len(self._ingester.documents) == 0:
                logger.warning("Index exists but contains no documents. Rebuilding...")
                # Try to rebuild if movies.csv exists
                movies_path = Path("data/movies.csv")
                if movies_path.exists():
                    return self.rebuild_index(str(movies_path)) > 0
                else:
                    logger.error(
                        f"movies.csv not found at {movies_path}. Cannot rebuild index."
                    )
                    return False

        logger.warning(f"No existing index found at {self.index_path}.")
        logger.warning("To create the index, run: make ingest")
        logger.warning(
            "Or: python -c 'from swipeflix.rag.ingest import ingest_movies; ingest_movies()'"
        )

        # Try to auto-create if movies.csv exists
        movies_path = Path("data/movies.csv")
        if movies_path.exists():
            logger.info("Found movies.csv. Attempting to create index automatically...")
            try:
                count = self.rebuild_index(str(movies_path))
                if count > 0:
                    logger.info(f"Successfully created index with {count} documents")
                    return True
            except Exception as e:
                logger.error(f"Failed to auto-create index: {e}")

        return False

    def rebuild_index(self, source_path: str = "data/movies.csv") -> int:
        """Rebuild the index from source data."""
        if not self._ingester:
            self._ingester = DocumentIngester(index_path=str(self.index_path))

        count = self._ingester.ingest(source_path, source_type="csv")
        logger.info(f"Rebuilt index with {count} documents")
        return count

    def is_ready(self) -> bool:
        """Check if the retriever is ready for queries."""
        return (
            self._ingester is not None
            and self._ingester.index is not None
            and len(self._ingester.documents) > 0
        )

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query string to embedding."""
        if self._embedding_model is None:
            # Simple hash-based embedding fallback (deterministic)
            hash_obj = hashlib.md5(query.lower().encode())
            hash_bytes = hash_obj.digest()
            vec = np.frombuffer(
                hash_bytes * (self.embedding_dim // 16 + 1), dtype=np.uint8
            )[: self.embedding_dim]
            vec = (vec.astype(np.float32) / 127.5) - 1.0
            return vec.reshape(1, -1).astype("float32")

        embedding = self._embedding_model.encode([query], convert_to_numpy=True)
        return embedding.astype("float32")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MovieDocument, float]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query string
            top_k: Number of documents to retrieve
            filters: Optional filters (e.g., {"genre": "Action", "year_min": 2010})

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not self.is_ready():
            logger.warning("Retriever not ready. No documents indexed.")
            return []

        start_time = time.time()
        k = top_k or self.top_k_default

        # Encode query
        query_embedding = self._encode_query(query)

        # Normalize for cosine similarity
        if FAISS_AVAILABLE:
            faiss.normalize_L2(query_embedding)

        # Search index
        # Retrieve more candidates for filtering
        search_k = k * 3 if filters else k

        # Check if index is valid
        if self._ingester.index is None:
            logger.error("FAISS index is None. Index may not be loaded correctly.")
            return []

        if self._ingester.index.ntotal == 0:
            logger.error("FAISS index is empty. Rebuild the index with: make ingest")
            return []

        try:
            distances, indices = self._ingester.index.search(query_embedding, search_k)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            logger.error(
                f"Index ntotal: {self._ingester.index.ntotal}, Query embedding shape: {query_embedding.shape}"
            )
            return []

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._ingester.documents):
                continue

            doc = self._ingester.documents[idx]
            distance = float(distances[0][i])

            # Convert FAISS distance to similarity score
            # FAISS returns squared L2 distances (lower is better)
            # For normalized embeddings (L2 normalized), distance = 2 - 2*cosine_similarity
            # So: cosine_similarity = 1 - (distance / 2)
            # Distance range: 0 (identical) to 2 (opposite)
            if distance <= 2.0:
                similarity_score = 1.0 - (distance / 2.0)
            else:
                # For non-normalized or very different vectors
                similarity_score = max(0.0, 1.0 / (1.0 + distance))

            # Apply score threshold (now using similarity, higher is better)
            if similarity_score < self.min_score_threshold:
                logger.debug(
                    f"Document {idx} filtered: similarity {similarity_score:.3f} < threshold {self.min_score_threshold}"
                )
                continue

            # Apply filters
            if filters and not self._matches_filters(doc, filters):
                continue

            # Store with similarity score (higher is better)
            results.append((doc, similarity_score))

            # Record relevance score metric (use similarity)
            rag_relevance_score.observe(similarity_score)

            if len(results) >= k:
                break

        # Record metrics
        duration = time.time() - start_time
        rag_retrieval_duration.observe(duration)
        rag_documents_retrieved.observe(len(results))

        logger.debug(
            f"Retrieved {len(results)} documents for '{query[:50]}...' "
            f"in {duration*1000:.1f}ms"
        )

        return results

    def _matches_filters(self, doc: MovieDocument, filters: Dict[str, Any]) -> bool:
        """Check if document matches all filters."""
        for key, value in filters.items():
            if key == "genre":
                if value.lower() not in doc.genre.lower():
                    return False
            elif key == "year_min":
                if doc.year < value:
                    return False
            elif key == "year_max":
                if doc.year > value:
                    return False
            elif key == "rating_min":
                if doc.rating < value:
                    return False
            elif key == "director":
                if value.lower() not in doc.director.lower():
                    return False

        return True

    def retrieve_by_id(self, doc_id: str) -> Optional[MovieDocument]:
        """Retrieve a specific document by ID."""
        if not self._ingester:
            return None

        idx = self._ingester.id_to_idx.get(doc_id)
        if idx is not None and idx < len(self._ingester.documents):
            return self._ingester.documents[idx]
        return None

    def get_similar_movies(
        self,
        movie_id: str,
        top_k: int = 5,
    ) -> List[Tuple[MovieDocument, float]]:
        """Find movies similar to a given movie."""
        doc = self.retrieve_by_id(movie_id)
        if not doc:
            return []

        # Use the movie's content as query
        return self.retrieve(doc.content, top_k=top_k + 1)[1:]  # Exclude self

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        keyword_weight: float = 0.3,
    ) -> List[Tuple[MovieDocument, float]]:
        """Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            top_k: Number of results
            keyword_weight: Weight for keyword matching (0-1)

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self.top_k_default

        # Get semantic results
        semantic_results = self.retrieve(query, top_k=k * 2)

        # Score documents with keyword matching
        query_terms = set(query.lower().split())
        scored_results = []

        for doc, semantic_score in semantic_results:
            # Compute keyword score
            doc_text = f"{doc.title} {doc.genre} {doc.director} {doc.plot}".lower()
            keyword_matches = sum(1 for term in query_terms if term in doc_text)
            keyword_score = keyword_matches / max(len(query_terms), 1)

            # Combine scores
            final_score = (
                1 - keyword_weight
            ) * semantic_score + keyword_weight * keyword_score
            scored_results.append((doc, final_score))

        # Sort and return top_k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]

    def get_context_for_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get context documents formatted for LLM prompts.

        Args:
            query: User query
            top_k: Number of documents

        Returns:
            List of document dicts ready for prompt formatting
        """
        results = self.retrieve(query, top_k=top_k)

        context = []
        for doc, score in results:
            context.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "genre": doc.genre,
                    "year": doc.year,
                    "rating": doc.rating,
                    "director": doc.director,
                    "cast": doc.cast,
                    "plot": doc.plot,
                    "relevance_score": score,
                }
            )

        return context


# Global retriever instance
_retriever: Optional[MovieRetriever] = None


def get_retriever() -> MovieRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = MovieRetriever()
    return _retriever
