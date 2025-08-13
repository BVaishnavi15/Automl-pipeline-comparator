import pandas as pd

# Load your results CSV
df = pd.read_csv("results.csv")

# Compute metrics
num_datasets = df['dataset'].nunique()
num_runs = len(df)

# Print results
print(f"Number of unique datasets tested: {num_datasets}")
print(f"Total number of experiments/run logs: {num_runs}")
