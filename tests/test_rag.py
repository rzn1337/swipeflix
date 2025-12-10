"""Tests for RAG pipeline components."""

import tempfile

import pytest


class TestMovieDocument:
    """Tests for MovieDocument dataclass."""

    def test_document_creation(self):
        """Test creating a movie document."""
        from swipeflix.rag.ingest import MovieDocument

        doc = MovieDocument(
            id="123",
            title="Avatar",
            genre="Action|Sci-Fi",
            year=2009,
            rating=7.2,
            director="James Cameron",
            plot="A marine on an alien planet",
        )

        assert doc.id == "123"
        assert doc.title == "Avatar"
        assert doc.year == 2009

    def test_document_to_dict(self):
        """Test converting document to dictionary."""
        from swipeflix.rag.ingest import MovieDocument

        doc = MovieDocument(
            id="123",
            title="Avatar",
            genre="Action",
        )

        d = doc.to_dict()
        assert d["id"] == "123"
        assert d["title"] == "Avatar"
        assert "metadata" in d

    def test_document_from_dict(self):
        """Test creating document from dictionary."""
        from swipeflix.rag.ingest import MovieDocument

        data = {
            "id": "456",
            "title": "Titanic",
            "year": 1997,
            "rating": 7.5,
        }

        doc = MovieDocument.from_dict(data)
        assert doc.id == "456"
        assert doc.title == "Titanic"
        assert doc.year == 1997


class TestDocumentIngester:
    """Tests for DocumentIngester."""

    def test_ingester_initialization(self):
        """Test ingester initialization."""
        from swipeflix.rag.ingest import DocumentIngester

        with tempfile.TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(index_path=tmpdir)

            assert ingester.embedding_dim > 0
            assert ingester.documents == []

    def test_load_movies_csv(self, sample_movies_csv):
        """Test loading movies from CSV."""
        from swipeflix.rag.ingest import DocumentIngester

        with tempfile.TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(index_path=tmpdir)
            docs = ingester.load_movies_csv(str(sample_movies_csv))

            assert len(docs) > 0
            assert all(hasattr(d, "title") for d in docs)
            assert all(hasattr(d, "content") for d in docs)

    def test_create_embeddings(self, sample_movies_csv):
        """Test embedding creation."""
        from swipeflix.rag.ingest import DocumentIngester

        with tempfile.TemporaryDirectory() as tmpdir:
            ingester = DocumentIngester(index_path=tmpdir)
            docs = ingester.load_movies_csv(str(sample_movies_csv))
            embeddings = ingester.create_embeddings(docs)

            assert embeddings.shape[0] == len(docs)
            assert embeddings.shape[1] == ingester.embedding_dim


class TestMovieRetriever:
    """Tests for MovieRetriever."""

    def test_retriever_not_ready_without_index(self):
        """Test retriever reports not ready without index."""
        from swipeflix.rag.retriever import MovieRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = MovieRetriever(index_path=tmpdir)
            assert not retriever.is_ready()

    def test_retrieve_returns_empty_when_not_ready(self):
        """Test retrieve returns empty when not ready."""
        from swipeflix.rag.retriever import MovieRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = MovieRetriever(index_path=tmpdir)
            results = retriever.retrieve("test query")

            assert results == []


class TestRAGGenerator:
    """Tests for RAGGenerator."""

    def test_generator_initialization(self):
        """Test RAG generator initialization."""
        # This may fail without proper dependencies
        try:
            from swipeflix.rag.generator import RAGGenerator

            generator = RAGGenerator()
            assert generator.max_context_docs > 0
        except ImportError:
            pytest.skip("RAG dependencies not available")


# Fixtures


@pytest.fixture
def sample_movies_csv(tmp_path):
    """Create a sample movies CSV file."""
    import pandas as pd

    movies = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "title": ["Avatar", "Titanic", "The Avengers"],
            "genre": ["Action|Sci-Fi", "Drama|Romance", "Action|Adventure"],
            "year": [2009, 1997, 2012],
            "rating": [7.2, 7.5, 7.4],
            "director": ["James Cameron", "James Cameron", "Joss Whedon"],
            "cast": ["[]", "[]", "[]"],
            "plot": [
                "A marine on an alien planet.",
                "A love story on a ship.",
                "Superheroes save the world.",
            ],
        }
    )

    csv_path = tmp_path / "movies.csv"
    movies.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_index(sample_movies_csv, tmp_path):
    """Create a sample FAISS index."""
    try:
        from swipeflix.rag.ingest import DocumentIngester

        index_path = tmp_path / "index"
        ingester = DocumentIngester(index_path=str(index_path))
        ingester.ingest(str(sample_movies_csv), source_type="csv")
        return index_path
    except ImportError:
        pytest.skip("FAISS not available")
