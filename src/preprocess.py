from __future__ import annotations

import pandas as pd


TARGET_MAP = {"No": 0, "Yes": 1}


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light type normalization used across train/inference."""
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


def infer_feature_types(
    df: pd.DataFrame,
    target_col: str = "Churn",
    categorical_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer categorical and numeric columns from a training frame."""
    id_cols = id_cols or ["id"]

    inferred_cats = set(df.select_dtypes(include=["object", "category"]).columns)
    if categorical_cols is not None:
        inferred_cats.update(categorical_cols)
    if "SeniorCitizen" in df.columns:
        inferred_cats.add("SeniorCitizen")

    excluded = set(id_cols + [target_col])
    categorical = [c for c in df.columns if c in inferred_cats and c not in excluded]
    numeric = [c for c in df.columns if c not in set(categorical) and c not in excluded]
    return categorical, numeric


def encode_target(y: pd.Series) -> pd.Series:
    """Convert target labels into 0/1 with validation."""
    mapped = y.map(TARGET_MAP)
    if mapped.isnull().any():
        raise ValueError("Target contains values outside {'No', 'Yes'}")
    return mapped.astype(int)


def make_features(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Backward-compatible one-shot feature generation."""
    out = normalize_frame(df)
    cols = [col for col in categorical_cols if col != "Churn"]
    cols = [col for col in cols if col in out.columns]
    out = pd.get_dummies(out, columns=cols, drop_first=True)
    if "Churn" in out.columns:
        out["Churn"] = encode_target(out["Churn"])
    return out


def prepare_fold_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    """Fit preprocessing on train fold only, then transform train/valid."""
    id_cols = id_cols or ["id"]
    tr = normalize_frame(train_df)
    va = normalize_frame(val_df)

    cats, nums = infer_feature_types(
        tr,
        target_col=target_col,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
    )

    medians = tr[nums].median().to_dict() if nums else {}
    for col, val in medians.items():
        tr[col] = tr[col].fillna(val)
        if col in va.columns:
            va[col] = va[col].fillna(val)

    x_tr_raw = tr.drop(columns=[target_col], errors="ignore")
    x_va_raw = va.drop(columns=[target_col], errors="ignore")

    cats_for_dummies = [c for c in cats if c in x_tr_raw.columns]
    x_tr = pd.get_dummies(x_tr_raw, columns=cats_for_dummies, drop_first=True)
    x_va = pd.get_dummies(x_va_raw, columns=cats_for_dummies, drop_first=True)
    x_va = x_va.reindex(columns=x_tr.columns, fill_value=0)

    y_tr = encode_target(tr[target_col])
    y_va = encode_target(va[target_col])

    artifacts = {
        "target_col": target_col,
        "id_cols": id_cols,
        "categorical_cols": cats,
        "numeric_cols": nums,
        "numeric_medians": medians,
        "feature_columns": list(x_tr.columns),
    }
    return x_tr, y_tr, x_va, y_va, artifacts


def prepare_full_training_features(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Create full training matrix and preprocessing artifacts."""
    id_cols = id_cols or ["id"]
    full = normalize_frame(df)

    cats, nums = infer_feature_types(
        full,
        target_col=target_col,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
    )

    medians = full[nums].median().to_dict() if nums else {}
    for col, val in medians.items():
        full[col] = full[col].fillna(val)

    x_raw = full.drop(columns=[target_col], errors="ignore")
    cats_for_dummies = [c for c in cats if c in x_raw.columns]
    x = pd.get_dummies(x_raw, columns=cats_for_dummies, drop_first=True)
    y = encode_target(full[target_col])

    artifacts = {
        "target_col": target_col,
        "id_cols": id_cols,
        "categorical_cols": cats,
        "numeric_cols": nums,
        "numeric_medians": medians,
        "feature_columns": list(x.columns),
    }
    return x, y, artifacts


def transform_with_artifacts(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Transform arbitrary frame into model feature space using saved artifacts."""
    out = normalize_frame(df)

    for col in artifacts.get("numeric_cols", []):
        if col in out.columns:
            out[col] = out[col].fillna(artifacts.get("numeric_medians", {}).get(col))

    x_raw = out.drop(columns=[artifacts.get("target_col", "Churn")], errors="ignore")
    cats_for_dummies = [
        c for c in artifacts.get("categorical_cols", []) if c in x_raw.columns
    ]
    x = pd.get_dummies(x_raw, columns=cats_for_dummies, drop_first=True)
    x = x.reindex(columns=artifacts.get("feature_columns", []), fill_value=0)
    return x