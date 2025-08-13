"""
H2O runner (init is handled in main)
- Auto-detects problem type
- Correctly treats target as factor (classification) or numeric (regression)
- Computes appropriate metrics
"""
import os
import time
from datetime import datetime

import pandas as pd
import h2o
from h2o.automl import H2OAutoML

from automl.utils import (
    detect_problem_type,
    remove_rare_classes,
    compute_classification_metrics,
    compute_regression_metrics,
)


def run_h2o(
    data_path: str,
    target_column: str,
    max_runtime_secs: int = 300,
    max_models: int | None = None,
    seed: int = 42,
    save_model: bool = False,
):
    ts = datetime.now().isoformat()
    t0 = time.time()

    # Load with pandas first to detect mode & pre-clean
    pdf = pd.read_csv(data_path)
    if target_column not in pdf.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    mode = detect_problem_type(pdf[target_column])

    dropped = []
    if mode == "classification":
        pdf, dropped = remove_rare_classes(pdf, target_column, min_count=2)
        if pdf[target_column].nunique() < 2:
            raise ValueError("Classification requires at least 2 classes with >=2 rows total.")

    # Send to H2O
    h2o_df = h2o.H2OFrame(pdf)

    # Cast target properly
    if mode == "classification":
        h2o_df[target_column] = h2o_df[target_column].asfactor()
    else:
        # ensure numeric; if not, try to coerce
        try:
            h2o_df[target_column] = h2o_df[target_column].asnumeric()
        except Exception:
            raise ValueError("Regression target must be numeric for H2O.")

    # 80/20 split
    train, test = h2o_df.split_frame(ratios=[0.8], seed=seed)

    x = [c for c in h2o_df.col_names if c != target_column]
    y = target_column

    aml_kwargs = {"max_runtime_secs": int(max_runtime_secs), "seed": int(seed)}
    if max_models is not None:
        aml_kwargs["max_models"] = int(max_models)

    print(f"🔧 H2O AutoML: max_runtime_secs={max_runtime_secs}, max_models={max_models}, mode={mode}")
    aml = H2OAutoML(**aml_kwargs)
    aml.train(x=x, y=y, training_frame=train)

    leader = aml.leader
    model_id = leader.model_id

    pred_hf = leader.predict(test)
    pred_pd = pred_hf.as_data_frame()

    # y_true from test frame
    y_true = test[y].as_data_frame()[y].values

    if mode == "classification":
        y_pred = pred_pd["predict"].values if "predict" in pred_pd.columns else None
        # probabilities: drop 'predict' column if present
        y_proba = None
        if "predict" in pred_pd.columns:
            prob_pd = pred_pd.drop(columns=["predict"])
            y_proba = prob_pd.values if prob_pd.shape[1] > 0 else None

        metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
        acc, f1, auc = metrics["accuracy"], metrics["f1"], metrics["auc"]
        rmse = mae = r2 = None
    else:
        # regression predicts a single 'predict' column
        y_pred = pred_pd["predict"].values if "predict" in pred_pd.columns else None
        metrics = compute_regression_metrics(y_true, y_pred)
        rmse, mae, r2 = metrics["rmse"], metrics["mae"], metrics["r2"]
        acc = f1 = auc = None

    # optionally save model
    save_path = None
    if save_model:
        save_dir = os.path.join("H2OModels", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        save_path = h2o.save_model(model=leader, path=save_dir, force=True)
        print(f"🔁 Saved H2O model to: {save_path}")

    t1 = time.time()

    autosettings = {
        "h2o_max_runtime_secs": max_runtime_secs,
        "h2o_max_models": max_models,
        "seed": seed,
        "h2o_model_path": save_path,
        "dropped_classes": dropped if dropped else None,
    }

    return {
        "timestamp": ts,
        "dataset": os.path.abspath(data_path),
        "framework": "H2O.ai",
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
