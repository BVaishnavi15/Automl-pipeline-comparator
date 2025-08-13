#!/usr/bin/env python3
"""
AutoML Pipeline Comparator — Universal (Classification + Regression)
"""
import argparse
import os
from datetime import datetime

from automl.autogluon_runner import run_autogluon
from automl.h2o_runner import run_h2o
from automl.utils import (
    ensure_results_csv,
    append_result,
    print_leaderboard,
)

SUPPORTED = ["autogluon", "h2o"]


def parse_args():
    p = argparse.ArgumentParser(description="AutoML Pipeline Comparator (Universal)")
    # Training / data
    p.add_argument("--data", help="Path to dataset CSV", required=False)
    p.add_argument("--target", help="Target/label column name", required=False)
    p.add_argument(
        "--frameworks",
        nargs="+",
        choices=SUPPORTED,
        default=SUPPORTED,
        help="Frameworks to run (space-separated). Default: autogluon h2o",
    )
    p.add_argument("--time_limit", type=int, default=300,
                   help="Per-framework time limit in seconds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # AutoGluon tuning
    p.add_argument("--ag_preset", type=str, default="medium",
                   help="AutoGluon preset: extreme|best|high|good|medium")
    p.add_argument("--ag_hpo", action="store_true",
                   help="Enable simple HPO (random search) for AutoGluon")

    # H2O tuning
    p.add_argument("--h2o_max_models", type=int, default=None,
                   help="H2O AutoML max_models (optional). If None, use max_runtime_secs only.")

    # Artifacts
    p.add_argument("--save_models", action="store_true",
                   help="Save model artifacts (AutoGluon folder / H2O exported model)")

    # Leaderboard
    p.add_argument("--leaderboard", action="store_true", help="Show leaderboard and exit")
    p.add_argument("--top", type=int, default=10, help="Top-N rows to show in leaderboard")
    p.add_argument("--mode", type=str, choices=["classification", "regression", "all"], default="all",
                   help="Leaderboard filter: classification|regression|all")

    return p.parse_args()


def main():
    args = parse_args()

    # Leaderboard only
    ensure_results_csv()
    if args.leaderboard:
        print_leaderboard(top_n=args.top, mode_filter=args.mode)
        return

    # Validate training inputs
    if not args.data or not args.target:
        raise SystemExit("Error: --data and --target are required (unless --leaderboard).")

    data_path = args.data
    target = args.target
    frameworks = [fw.lower() for fw in args.frameworks]

    # Initialize H2O once if included
    h2o_started = False
    if "h2o" in frameworks:
        try:
            import h2o
            print("🔁 Initializing H2O (once for this run)...")
            h2o.init()
            h2o_started = True
        except Exception as e:
            print("⚠️ Failed to initialize H2O:", e)
            print("Continuing with other frameworks (omit 'h2o' to skip).")
            frameworks = [fw for fw in frameworks if fw != "h2o"]

    # Run frameworks sequentially
    for fw in frameworks:
        print(f"\n=== Running framework: {fw} ===")
        try:
            if fw == "autogluon":
                result = run_autogluon(
                    data_path=data_path,
                    target_column=target,
                    time_limit=args.time_limit,
                    preset=args.ag_preset,
                    enable_hpo=args.ag_hpo,
                    seed=args.seed,
                    save_models=args.save_models,
                )
            elif fw == "h2o":
                result = run_h2o(
                    data_path=data_path,
                    target_column=target,
                    max_runtime_secs=args.time_limit,
                    max_models=args.h2o_max_models,
                    seed=args.seed,
                    save_model=args.save_models,
                )
            else:
                raise ValueError(f"Unsupported framework: {fw}")

            append_result(result)
            print("✅ Logged result:")
            print(result)

        except Exception as exc:
            error_row = {
                "timestamp": datetime.now().isoformat(),
                "dataset": os.path.abspath(data_path),
                "framework": fw,
                "model_id": None,
                "accuracy": None,
                "f1": None,
                "auc": None,
                "rmse": None,
                "mae": None,
                "r2": None,
                "train_time_sec": None,
                "mode": None,
                "autosettings": None,
                "notes": f"error: {exc}",
            }
            append_result(error_row)
            print(f"❌ Error running {fw}: {exc}")

    # Shutdown H2O once
    if h2o_started:
        try:
            import h2o
            print("\n🔁 Shutting down H2O...")
            h2o.shutdown(prompt=False)
        except Exception as e:
            print("⚠️ H2O shutdown failed:", e)


if __name__ == "__main__":
    main()
