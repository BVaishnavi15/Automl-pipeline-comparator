import os
import time
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor
from automl.utils import (
    detect_problem_type,
    remove_rare_classes,
    compute_classification_metrics,
    compute_regression_metrics,
)


def run_autogluon(
    data_path: str,
    target_column: str,
    time_limit: int = 300,
    preset: str = "medium",
    enable_hpo: bool = False,
    seed: int = 42,
    save_models: bool = False,
):
    ts = datetime.now().isoformat()
    t0 = time.time()

    df = pd.read_csv(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    mode = detect_problem_type(df[target_column])

    # Clean rare classes (classification only)
    dropped = []
    if mode == "classification":
        df, dropped = remove_rare_classes(df, target_column, min_count=2)

    # If dataset is too small / single-class
    if mode == "classification" and df[target_column].nunique() < 2:
        raise ValueError("Classification requires at least 2 classes with >=2 rows total.")

    # Split
    stratify = None
    if mode == "classification" and df[target_column].nunique() > 1:
        # try stratified split; fallback if it fails
        try:
            stratify = df[target_column]
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=seed, stratify=stratify
            )
        except Exception:
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=seed, stratify=None
            )
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)

    # Train
    hpo_args = None
    if enable_hpo:
        hpo_args = {"num_trials": 8, "search_strategy": "random"}

    save_path = None
    if save_models:
        save_path = os.path.join("AutogluonModels", f"ag-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_path, exist_ok=True)
        print(f"🔁 Saving AutoGluon models to {save_path}")

    # Let AutoGluon infer problem_type; it handles both modes well
    print(f"🔧 AutoGluon: preset={preset}, HPO={enable_hpo}, time_limit={time_limit}s, mode={mode}")
    predictor = TabularPredictor(label=target_column, path=save_path).fit(
        train_data=train_df,
        time_limit=time_limit,
        presets=preset,
        hyperparameter_tune_kwargs=hpo_args,
    )

    X_test = test_df.drop(columns=[target_column])
    y_true = test_df[target_column].values
    y_pred = predictor.predict(X_test)
    y_pred = y_pred.values if hasattr(y_pred, "values") else y_pred

    # probabilities for classification
    y_proba = None
    if mode == "classification":
        try:
            proba_df = predictor.predict_proba(X_test)
            y_proba = proba_df.values if hasattr(proba_df, "values") else None
        except Exception:
            y_proba = None

    # Metrics
    if mode == "classification":
        metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
        acc, f1, auc = metrics["accuracy"], metrics["f1"], metrics["auc"]
        rmse = mae = r2 = None
    else:
        metrics = compute_regression_metrics(y_true, y_pred)
        rmse, mae, r2 = metrics["rmse"], metrics["mae"], metrics["r2"]
        acc = f1 = auc = None

    # Best model id
    try:
        model_id = predictor.get_model_best()
    except Exception:
        model_id = None

    t1 = time.time()

    autosettings = {
        "preset": preset,
        "hpo_enabled": bool(enable_hpo),
        "seed": seed,
        "ag_model_path": save_path if save_models else None,
        "dropped_classes": dropped if dropped else None,
    }

    return {
        "timestamp": ts,
        "dataset": os.path.abspath(data_path),
        "framework": "AutoGluon",
        "model_id": model_id,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_time_sec": round(t1 - t0, 2),
        "mode": mode,
        "autosettings": str(autosettings),
        "notes": "" if not dropped else f"Removed rare classes: {dropped}",
    }
