## The AutoML Pipeline Comparator
A Python-based command-line tool to train, compare, and benchmark multiple AutoML frameworks (currently AutoGluon and H2O.ai) on your datasets — all from a single terminal command.
Designed for data scientists, ML engineers, and researchers who want a quick, reproducible, and standardized way to evaluate AutoML frameworks without switching environments, rewriting code, or manually comparing results.


### Project Overview
1. Runs multiple AutoML frameworks sequentially on the same dataset.
2. Detects problem type automatically (classification or regression).
3. Computes standardized evaluation metrics:
Classification: Accuracy, Weighted F1-score, AUC;
Regression: RMSE, MAE, R²
4. Logs results into a single CSV leaderboard (results.csv) with timestamps & metadata.
5. Handles rare-class splits without crashes.
6. Captures framework-specific tuning settings in logs.
7. Allows easy extension for future frameworks.


### Why Use This Tool
This tool automates comparisons and produces a clear leaderboard, saving hours of manual work. Real-world ML often requires comparing multiple AutoML frameworks to determine
1. Which framework works best for small vs large datasets.
2. How performance changes with different presets or hyperparameter tuning.
3. Which approach is reproducible and reliable under time constraints.


### Features
1. Multi-framework support — AutoGluon, H2O.ai (more coming soon: TPOT, Auto-sklearn, FLAML)
2. Universal task support — Classification & Regression
3. Automatic problem detection
4. Standardized metrics — Acc/F1/AUC or RMSE/MAE/R²
5. Time-limited training per framework
6. Batch execution — Run multiple frameworks in one go
7. Leaderboard mode — View ranked results from all runs
8. Results logging — Saves runs to results.csv for tracking over time
9. Optimized execution:
No temp files for AutoGluon (uses in-memory DataFrames),
Avoids repeated startup/shutdown for H2O.ai
10. Plug-and-play architecture — Easily add more frameworks
11. Parallel execution support
12. Optional:
Hyperparameter tuning (--ag_hpo)
Model saving
Regression metric support

### Technologies Used
| Tool / Library   | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| **Python 3.x**   | Core programming language                             |
| **AutoGluon**    | AutoML training for tabular datasets                  |
| **H2O.ai**       | AutoML training for tabular datasets                  |
| **pandas**       | Data manipulation & CSV logging                       |
| **scikit-learn** | Metric computation (Accuracy, F1, AUC, RMSE, MAE, R²) |
| **tabulate**     | Pretty leaderboard tables                             |
| **argparse**     | CLI parsing                                           |
| **CSV**          | Persistent benchmark results                          |


### Terminal Commands Usage
* python main.py --data data/train.csv --target target_column--frameworks autogluon h2o --time_limit 300--ag_preset best --ag_hpo--save_models
###### --data: Path to dataset CSV
###### --target: Target/label column
###### --frameworks: Space-separated frameworks to run (default: all supported)
###### --time_limit: Seconds per framework
###### --ag_preset: AutoGluon preset (extreme|best|high|good|medium)
###### --ag_hpo: Enable AutoGluon hyperparameter tuning
###### --save_models: Save model artifacts to disk

1. python main.py --frameworks autogluon h2o --data "C:/Users/Vaishnavi B/OneDrive/Desktop/AutoML/automl-pipeline-comparator/data/mydataset.csv" --target Survived

2. python main.py --frameworks autogluon h2o --data "C:/Users/Vaishnavi B/OneDrive/Desktop/AutoML/automl-pipeline-comparator/data/newdata.csv" --target "Caloires Required"



1. (Train Models)
python main.py --train path/to/train.csv --label target_column --frameworks autogluon h2o --time_limit 300
2. (Show Leaderboard)
python main.py --leaderboard
3. (Evaluate on Test Dataset(Load an already trained model and evaluate it on a new dataset.))=
python main.py --evaluate path/to/test.csv --label target_column --framework autogluon
4. (Cross-Validation(Run k-fold cross-validation.))=
python main.py --train data/train.csv --label target --frameworks autogluon h2o --cv 5
5. (Export Best Model(Save the best-performing model for deployment.))=
python main.py --export best_model --framework autogluon
6. (Detailed Run Log)=
python main.py --log_run 2025-08-12T18:02:03
7. (Delete Results)=
python main.py --clear_results
python main.py --delete_run 2025-08-12T18:02:03
8. (Dataset Summary)=
python main.py --summary data/train.csv
9. (Parallel Execution)=
python main.py --train data/train.csv --label target --frameworks autogluon h2o --time_limit 300 --parallel


































