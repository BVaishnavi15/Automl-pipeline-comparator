import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except Exception:
    _HAS_TABULATE = False

RESULTS_CSV = "results.csv"
RESULTS_COLUMNS = [
    "timestamp", "dataset", "framework", "model_id",
    "accuracy", "f1", "auc",         # classification
    "rmse", "mae", "r2",             # regression
    "train_time_sec", "mode", "autosettings", "notes"
]

# ---------- Problem-type detection & sanitization ----------

def detect_problem_type(series: pd.Series, multiclass_threshold: int = 20) -> str:
    """
    Heuristic:
    - If numeric with unique values > threshold => regression
    - Otherwise => classification
    """
    s = series.dropna()
    nunique = s.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(s)
    if is_numeric and nunique > multiclass_threshold:
        return "regression"
    return "classification"


def remove_rare_classes(df: pd.DataFrame, target: str, min_count: int = 2):
    """
    Removes rows where target class frequency < min_count.
    Returns a (df_clean, dropped_classes_list) tuple.
    """
    vc = df[target].value_counts(dropna=False)
    rare = vc[vc < min_count].index.tolist()
    if len(rare) == 0:
        return df, []
    df_clean = df[~df[target].isin(rare)].copy()
    return df_clean, rare


# ---------- Metrics ----------

def compute_classification_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique = np.unique(y_true)
    average = "binary" if unique.size == 2 else "macro"

    # Accuracy & F1
    try:
        accuracy = float(round(accuracy_score(y_true, y_pred), 6))
    except Exception:
        accuracy = None
    try:
        f1 = float(round(f1_score(y_true, y_pred, average=average, zero_division=0), 6))
    except Exception:
        f1 = None

    # AUC (best-effort)
    auc_val = None
    try:
        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            if y_proba.ndim == 1:
                auc_val = float(round(roc_auc_score(y_true, y_proba), 6))
            elif y_proba.ndim == 2:
                if y_proba.shape[1] == 2:
                    auc_val = float(round(roc_auc_score(y_true, y_proba[:, 1]), 6))
                else:
                    auc_val = float(round(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"), 6
                    ))
    except Exception:
        auc_val = None

    return {"accuracy": accuracy, "f1": f1, "auc": auc_val}


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    try:
        rmse = float(round(np.sqrt(mean_squared_error(y_true, y_pred)), 6))
    except Exception:
        rmse = None
    try:
        mae = float(round(mean_absolute_error(y_true, y_pred), 6))
    except Exception:
        mae = None
    try:
        r2 = float(round(r2_score(y_true, y_pred), 6))
    except Exception:
        r2 = None
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------- Results CSV ----------

def ensure_results_csv():
    if not os.path.exists(RESULTS_CSV):
        df = pd.DataFrame(columns=RESULTS_COLUMNS)
        df.to_csv(RESULTS_CSV, index=False)


def append_result(row: dict):
    """
    Normalize the row to RESULTS_COLUMNS and append to CSV.
    """
    ensure_results_csv()
    std = {k: None for k in RESULTS_COLUMNS}
    std.update(row)

    if std.get("timestamp") is None:
        std["timestamp"] = datetime.now().isoformat()

    if std.get("dataset"):
        std["dataset"] = os.path.abspath(str(std["dataset"]))

    df = pd.read_csv(RESULTS_CSV)
    df = pd.concat([df, pd.DataFrame([std])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"📥 Appended run to {RESULTS_CSV}")


# ---------- Leaderboard ----------

def _sort_for_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "classification":
        return df.sort_values(by=["f1", "accuracy", "auc"], ascending=[False, False, False])
    else:  # regression
        # higher r2 is better, lower rmse/mae are better
        return df.sort_values(by=["r2", "rmse", "mae"], ascending=[False, True, True])


def print_leaderboard(top_n=10, mode_filter: str = "all"):
    ensure_results_csv()
    df = pd.read_csv(RESULTS_CSV)

    if df.empty:
        print("No runs logged yet.")
        return

    # Make numeric where possible
    for c in ["accuracy", "f1", "auc", "rmse", "mae", "r2", "train_time_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter by mode if requested
    if mode_filter in ("classification", "regression"):
        subset = df[df["mode"] == mode_filter].copy()
        if subset.empty:
            print(f"No runs for mode='{mode_filter}'.")
            return
        subset = _sort_for_mode(subset, mode_filter).head(top_n)
        _print_table(subset, title=f"🏆 Leaderboard — {mode_filter} (top {top_n})")
        return

    # Otherwise, show both if available
    df_cls = df[df["mode"] == "classification"].copy()
    df_reg = df[df["mode"] == "regression"].copy()

    if not df_cls.empty:
        cls_sorted = _sort_for_mode(df_cls, "classification").head(top_n)
        _print_table(cls_sorted, title=f"🏆 Leaderboard — classification (top {top_n})")

    if not df_reg.empty:
        reg_sorted = _sort_for_mode(df_reg, "regression").head(top_n)
        _print_table(reg_sorted, title=f"🏆 Leaderboard — regression (top {top_n})")

    if df_cls.empty and df_reg.empty:
        print("No valid runs to show.")


def _print_table(df: pd.DataFrame, title: str):
    cols = [
        "timestamp", "mode", "framework", "dataset", "model_id",
        "accuracy", "f1", "auc", "rmse", "mae", "r2",
        "train_time_sec", "autosettings", "notes"
    ]
    df_disp = df[cols].copy()
    print("\n" + title)
    if _HAS_TABULATE:
        print(tabulate(df_disp, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".6f"))
    else:
        print(df_disp.to_string(index=False))
