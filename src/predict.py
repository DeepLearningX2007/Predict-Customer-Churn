import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import transform_with_artifacts


def predict_from_dataframe(df: pd.DataFrame, model_dir: str | Path, threshold: float = 0.5) -> pd.DataFrame:
    """Load model artifacts and return probability/label predictions."""
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file was not found: {metadata_path}")

    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    artifacts = metadata["preprocessing"]
    X = transform_with_artifacts(df, artifacts)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "churn_proba": proba,
            "churn_pred": pred,
        }
    )

    for col in artifacts.get("id_cols", []):
        if col in df.columns:
            output.insert(0, col, df[col].values)
            break

    return output


def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved churn model.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with model.pkl and metadata.json")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save prediction CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    pred_df = predict_from_dataframe(df, model_dir=args.model_dir, threshold=args.threshold)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
