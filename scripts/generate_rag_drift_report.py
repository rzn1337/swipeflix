"""Generate Evidently drift report for RAG retrieval corpus.

Monitors data drift in the movie document corpus used for RAG retrieval.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

try:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not available. Install with: pip install evidently")


def load_reference_corpus(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load reference corpus (baseline movie data)."""
    movies_file = data_dir / "movies.csv"

    if not movies_file.exists():
        raise FileNotFoundError(f"Reference data not found: {movies_file}")

    df = pd.read_csv(movies_file)
    logger.info(f"Loaded reference corpus: {len(df)} movies")

    # Select relevant features for drift monitoring
    features = ["genre", "year", "rating", "director"]
    available_features = [f for f in features if f in df.columns]

    return df[available_features + ["id", "title"]]


def load_current_corpus(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load current corpus (can be same as reference for demo)."""
    # In production, this would load recent/updated data
    # For demo, we use the same data but could add noise/variation
    return load_reference_corpus(data_dir)


def generate_rag_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path = Path("monitoring/evidently"),
) -> Path:
    """Generate Evidently drift report for RAG corpus."""
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently not available. Cannot generate report.")
        return None

    logger.info("Generating RAG corpus drift report...")

    # Create report
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    # Run report
    report.run(reference_data=reference_data, current_data=current_data)

    # Save HTML report
    output_path.mkdir(parents=True, exist_ok=True)
    html_path = output_path / "rag_corpus_drift_report.html"
    report.save_html(str(html_path))
    logger.info(f"✅ Report saved to: {html_path}")

    # Save JSON
    json_path = output_path / "rag_corpus_drift_report.json"
    report.save_json(str(json_path))
    logger.info(f"✅ JSON report saved to: {json_path}")

    # Extract and print summary
    report_dict = report.as_dict()

    logger.info("=" * 60)
    logger.info("RAG CORPUS DRIFT REPORT SUMMARY")
    logger.info("=" * 60)

    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            if "result" in metric:
                result = metric["result"]
                if "drift_score" in result:
                    drift_score = result["drift_score"]
                    logger.info(f"📊 Drift Score: {drift_score:.4f}")

                    if drift_score > 0.15:
                        logger.warning(
                            "⚠️  HIGH DRIFT in retrieval corpus! Consider updating index."
                        )
                    elif drift_score > 0.05:
                        logger.info("⚡ Moderate drift detected. Monitor closely.")
                    else:
                        logger.info("✅ Low drift. Corpus distribution stable.")

                # Print feature-level drift
                if "column_name" in result and "drift_detected" in result:
                    if result["drift_detected"]:
                        logger.warning(f"  - {result['column_name']}: Drift detected")
    except Exception as e:
        logger.warning(f"Could not extract drift metrics: {e}")

    logger.info("=" * 60)

    return html_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAG corpus drift report")
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Directory containing movie data",
    )
    parser.add_argument(
        "--output",
        default="monitoring/evidently",
        type=Path,
        help="Output directory for reports",
    )

    args = parser.parse_args()

    try:
        # Load data
        reference_data = load_reference_corpus(args.data_dir)
        current_data = load_current_corpus(args.data_dir)

        # Generate report
        report_path = generate_rag_drift_report(
            reference_data=reference_data,
            current_data=current_data,
            output_path=args.output,
        )

        if report_path:
            print("\n✅ RAG corpus drift report generated!")
            print(f"📄 View at: {report_path}")
            print("🌐 Or access Evidently service: http://localhost:7000")

    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
