"""
RAG Document Ingestion Entry Point

This module provides a top-level entry point for the RAG ingestion pipeline.
It wraps the main ingestion functionality from swipeflix.rag.ingest.

Usage:
    python src/ingest.py --source data/movies.csv --index-path data/faiss_index
    python -m src.ingest --source data/movies.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from swipeflix.rag.ingest import DocumentIngester  # noqa: E402


def main():
    """Main entry point for ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents for RAG pipeline")
    parser.add_argument(
        "--source",
        default="data/movies.csv",
        help="Path to source data file (CSV or JSON)",
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

    # Create ingester and run
    ingester = DocumentIngester(index_path=args.index_path)
    count = ingester.ingest(args.source, source_type=args.source_type)

    print(f"✅ Successfully ingested {count} documents")
    print(f"📁 Index saved to: {args.index_path}")


if __name__ == "__main__":
    main()
