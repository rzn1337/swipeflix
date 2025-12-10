"""Document ingestion pipeline for SwipeFlix RAG.

This module handles:
- Loading movie data from CSV/JSON sources
- Creating document embeddings using sentence-transformers
- Building and persisting FAISS vector index
"""

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    SBERT_AVAILABLE = False
    logger.warning(f"sentence-transformers not available: {e}")
    logger.warning("Falling back to simple keyword-based embeddings")


@dataclass
class MovieDocument:
    """A movie document for the RAG system."""

    id: str
    title: str
    genre: str = ""
    year: int = 0
    rating: float = 0.0
    director: str = ""
    cast: str = ""
    plot: str = ""
    content: str = ""  # Combined searchable content
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "genre": self.genre,
            "year": self.year,
            "rating": self.rating,
            "director": self.director,
            "cast": self.cast,
            "plot": self.plot,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MovieDocument":
        return cls(
            id=str(data.get("id", "")),
            title=data.get("title", ""),
            genre=data.get("genre", ""),
            year=int(data.get("year", 0) or 0),
            rating=float(data.get("rating", 0) or 0),
            director=data.get("director", ""),
            cast=str(data.get("cast", "")),
            plot=data.get("plot", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )


class DocumentIngester:
    """Ingestion pipeline for movie documents.

    Handles loading, processing, embedding, and indexing of movie data
    for efficient retrieval in the RAG pipeline.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: str = "data/faiss_index",
        chunk_size: int = 512,
    ):
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        self.chunk_size = chunk_size

        # Initialize embedding model
        if SBERT_AVAILABLE:
            try:
                logger.info(f"Loading embedding model: {embedding_model}")
                self._embedding_model = SentenceTransformer(embedding_model)
                self.embedding_dim = (
                    self._embedding_model.get_sentence_embedding_dimension()
                )
            except (OSError, RuntimeError) as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.warning("Falling back to simple embeddings")
                self._embedding_model = None
                self.embedding_dim = 384
        else:
            self._embedding_model = None
            self.embedding_dim = 384  # Default for MiniLM

        # Documents and index
        self.documents: List[MovieDocument] = []
        self.index: Optional[faiss.Index] = None
        self.id_to_idx: Dict[str, int] = {}

    def load_movies_csv(self, filepath: str) -> List[MovieDocument]:
        """Load movies from CSV file."""
        logger.info(f"Loading movies from: {filepath}")

        df = pd.read_csv(filepath)
        documents = []

        for _, row in df.iterrows():
            # Create combined content for embedding
            content_parts = [
                f"Title: {row.get('title', '')}",
                f"Genre: {row.get('genre', '')}",
                f"Year: {row.get('year', '')}",
                f"Director: {row.get('director', '')}",
                f"Plot: {row.get('plot', '')}",
            ]
            content = " | ".join(content_parts)

            doc = MovieDocument(
                id=str(row.get("id", "")),
                title=row.get("title", ""),
                genre=row.get("genre", ""),
                year=int(row.get("year", 0) or 0),
                rating=float(row.get("rating", 0) or 0),
                director=row.get("director", ""),
                cast=str(row.get("cast", "")),
                plot=row.get("plot", ""),
                content=content,
                metadata={
                    "budget": row.get("budget"),
                    "revenue": row.get("revenue"),
                    "runtime": row.get("runtime"),
                    "popularity": row.get("popularity"),
                },
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} movie documents")
        return documents

    def load_documents_json(self, filepath: str) -> List[MovieDocument]:
        """Load documents from JSON file."""
        logger.info(f"Loading documents from: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        documents = [MovieDocument.from_dict(d) for d in data]
        logger.info(f"Loaded {len(documents)} documents from JSON")
        return documents

    def create_embeddings(self, documents: List[MovieDocument]) -> np.ndarray:
        """Create embeddings for documents."""
        if not self._embedding_model:
            logger.warning(
                "No embedding model available. Using simple TF-IDF-like embeddings."
            )
            # Create simple embeddings based on text content
            # This is a fallback when sentence-transformers isn't available
            embeddings = []
            for doc in documents:
                # Simple hash-based embedding (deterministic)
                text = doc.content.lower()
                # Create 384-dim vector from text hash
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                # Repeat hash to get 384 dimensions
                vec = np.frombuffer(hash_bytes * (384 // 16 + 1), dtype=np.uint8)[:384]
                # Normalize to [-1, 1]
                vec = (vec.astype(np.float32) / 127.5) - 1.0
                embeddings.append(vec)
            return np.array(embeddings).astype("float32")

        logger.info(f"Creating embeddings for {len(documents)} documents...")

        # Extract content for embedding
        texts = [doc.content for doc in documents]

        # Batch encode
        embeddings = self._embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=64,
        )

        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings.astype("float32")

    def build_index(self, documents: List[MovieDocument]) -> None:
        """Build FAISS index from documents."""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Cannot build index.")
            return

        self.documents = documents

        # Create embeddings
        embeddings = self.create_embeddings(documents)

        # Build FAISS index
        logger.info("Building FAISS index...")

        # Use IVF index for larger datasets, flat for small
        if len(documents) > 10000:
            # IVF index with clustering
            nlist = min(100, len(documents) // 100)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
        else:
            # Simple flat index for smaller datasets
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Add embeddings to index
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Build ID mapping
        self.id_to_idx = {doc.id: i for i, doc in enumerate(documents)}

        logger.info(f"Built index with {self.index.ntotal} vectors")

    def save_index(self, path: Optional[str] = None) -> None:
        """Save index and documents to disk."""
        save_path = Path(path) if path else self.index_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index and FAISS_AVAILABLE:
            index_file = save_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            logger.info(f"Saved FAISS index to: {index_file}")

        # Save documents
        docs_file = save_path / "documents.pkl"
        with open(docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved {len(self.documents)} documents to: {docs_file}")

        # Save ID mapping
        mapping_file = save_path / "id_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(self.id_to_idx, f)

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load index and documents from disk."""
        load_path = Path(path) if path else self.index_path

        try:
            # Load FAISS index
            if FAISS_AVAILABLE:
                index_file = load_path / "index.faiss"
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

            # Load documents
            docs_file = load_path / "documents.pkl"
            if docs_file.exists():
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")

            # Load ID mapping
            mapping_file = load_path / "id_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, "r") as f:
                    self.id_to_idx = json.load(f)

            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def ingest(
        self,
        source: Union[str, List[MovieDocument]],
        source_type: str = "csv",
    ) -> int:
        """Run the full ingestion pipeline.

        Args:
            source: Path to data file or list of documents
            source_type: "csv", "json", or "documents"

        Returns:
            Number of documents ingested
        """
        # Load documents
        if isinstance(source, list):
            documents = source
        elif source_type == "csv":
            documents = self.load_movies_csv(source)
        elif source_type == "json":
            documents = self.load_documents_json(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Build index
        self.build_index(documents)

        # Save to disk
        self.save_index()

        logger.info(f"Ingestion complete: {len(documents)} documents indexed")
        return len(documents)


def ingest_movies(
    movies_path: str = "data/movies.csv",
    index_path: str = "data/faiss_index",
) -> DocumentIngester:
    """Convenience function to ingest movie data.

    Args:
        movies_path: Path to movies CSV file
        index_path: Path to save FAISS index

    Returns:
        Configured DocumentIngester instance
    """
    ingester = DocumentIngester(index_path=index_path)
    ingester.ingest(movies_path, source_type="csv")
    return ingester


if __name__ == "__main__":
    # CLI for running ingestion
    import argparse

    parser = argparse.ArgumentParser(description="Ingest movie data for RAG")
    parser.add_argument(
        "--source",
        default="data/movies.csv",
        help="Path to source data file",
    )
    parser.add_argument(
        "--index-path",
        default="data/faiss_index",
        help="Path to save FAISS index",
    )
    parser.add_argument(
        "--source-type",
        default="csv",
        choices=["csv", "json"],
        help="Source file type",
    )

    args = parser.parse_args()

    ingester = DocumentIngester(index_path=args.index_path)
    count = ingester.ingest(args.source, source_type=args.source_type)
    print(f"Successfully ingested {count} documents")
