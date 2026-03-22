import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.preprocess import prepare_fold_features, prepare_full_training_features


def _build_model(model_type, params):
    if model_type == "lgb":
        return LGBMClassifier(**params)
    if model_type == "xgb":
        return XGBClassifier(**params)
    if model_type == "cat":
        return CatBoostClassifier(**params)
    raise ValueError("model_type must be 'lgb', 'xgb', or 'cat'")


def run_cv(
    df,
    target_col,
    model_type,
    params,
    n_splits=4,
    seed=42,
    categorical_cols=None,
    id_cols=None,
    return_oof_frame=False,
    threshold=0.5,
):

    X = df.drop(target_col, axis=1, errors="ignore")
    y = df[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(df))
    y_true = np.zeros(len(df), dtype=int)
    fold_ids = np.full(len(df), -1, dtype=int)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}")

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        X_train, y_train, X_val, y_val, _ = prepare_fold_features(
            train_df,
            val_df,
            target_col=target_col,
            categorical_cols=categorical_cols,
            id_cols=id_cols,
        )

        model = _build_model(model_type, params)

        if model_type == "lgb":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

        elif model_type == "xgb":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False,
            )

        elif model_type == "cat":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False,
            )

        y_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = y_pred
        y_true[val_idx] = y_val.to_numpy()
        fold_ids[val_idx] = fold + 1

        auc = roc_auc_score(y_val, y_pred)
        print(f"AUC: {auc:.4f}")
        metrics.append(auc)

    print(f"\nCV AUC: {np.mean(metrics):.4f}")

    if return_oof_frame:
        oof_df = pd.DataFrame(
            {
                "row_index": np.arange(len(df)),
                "fold": fold_ids,
                "y_true": y_true,
                "oof_pred": oof,
            }
        )

        for col in id_cols or ["id"]:
            if col in df.columns:
                oof_df.insert(0, col, df[col].values)
                break

        oof_df["oof_pred_label"] = (oof_df["oof_pred"] >= threshold).astype(int)
        return oof, metrics, oof_df

    return oof, metrics


def train_and_save(
    df: pd.DataFrame,
    target_col: str,
    model_type: str,
    params: dict,
    output_dir: str | Path,
    categorical_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
):
    """Train on full data and save both model and preprocessing artifacts."""
    model = _build_model(model_type, params)
    X_train, y_train, artifacts = prepare_full_training_features(
        df,
        target_col=target_col,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
    )

    if model_type == "cat":
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.pkl"
    metadata_path = save_dir / "metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "model_type": model_type,
        "params": params,
        "preprocessing": artifacts,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return model, metadata