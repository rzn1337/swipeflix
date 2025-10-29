"""Generate Evidently data drift report for SwipeFlix."""

import argparse
from pathlib import Path

import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from loguru import logger


def load_data(data_dir: Path):
    """Load reference and current data."""
    ratings_path = data_dir / "ratings.csv"
    
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    
    # Load all data
    df = pd.read_csv(ratings_path)
    
    # Split into reference (older data) and current (recent data)
    # Use timestamp if available, otherwise split randomly
    split_idx = int(len(df) * 0.7)
    reference_data = df.iloc[:split_idx].copy()
    current_data = df.iloc[split_idx:].copy()
    
    logger.info(f"Reference data: {len(reference_data)} rows")
    logger.info(f"Current data: {len(current_data)} rows")
    
    return reference_data, current_data


def generate_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, output_path: Path):
    """Generate Evidently drift report."""
    logger.info("Generating Evidently drift report...")
    
    # Create report with multiple metric presets
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset() if 'rating' in reference_data.columns else DataQualityPreset(),
        ]
    )
    
    # Run report
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save HTML report
    html_path = output_path / "drift_report.html"
    report.save_html(str(html_path))
    logger.info(f"Report saved to: {html_path}")
    
    # Save JSON for programmatic access
    json_path = output_path / "drift_report.json"
    report.save_json(str(json_path))
    logger.info(f"JSON report saved to: {json_path}")
    
    # Extract key metrics
    report_dict = report.as_dict()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DRIFT REPORT SUMMARY")
    logger.info("=" * 60)
    
    try:
        # Try to extract drift metrics
        metrics = report_dict.get('metrics', [])
        for metric in metrics:
            if 'result' in metric and 'drift_score' in metric['result']:
                drift_score = metric['result']['drift_score']
                logger.info(f"Drift Score: {drift_score:.4f}")
                
                if drift_score > 0.15:
                    logger.warning("⚠️  HIGH DRIFT DETECTED! Consider retraining model.")
                elif drift_score > 0.05:
                    logger.info("⚡ Moderate drift detected. Monitor closely.")
                else:
                    logger.info("✅ Low drift. Data distribution stable.")
    except Exception as e:
        logger.warning(f"Could not extract drift score: {e}")
    
    logger.info("=" * 60)
    
    return html_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("monitoring/evidently"),
        help="Directory to save reports",
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    reference_data, current_data = load_data(args.data_dir)
    
    # Generate report
    report_path = generate_report(reference_data, current_data, args.output_dir)
    
    logger.info(f"✅ Report generation complete! Open: {report_path}")


if __name__ == "__main__":
    main()

