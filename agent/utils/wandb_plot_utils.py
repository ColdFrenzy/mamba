import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from agent.utils.paths import PLOT_DIR

# Use a non-interactive backend (important for headless)
matplotlib.use('Agg')

# Initialize API
api = wandb.Api()

# Load all runs (filter if needed)
runs = api.runs("francesco_diag/3m")

# Group runs by their names
grouped_data = {}

for run in runs:
    group_key = run.name  # grouping by run name

    try:
        history = run.history()  # Pulls full history
        # Make sure the needed columns exist
        if 'eval/win_rate' not in history.columns or 'eval/eval_steps' not in history.columns:
            print(f"Skipping {run.name}: Missing columns")
            continue

        df = history[['eval/eval_steps', 'eval/win_rate']].dropna()
        if df.empty:
            print(f"Skipping {run.name}: Empty DataFrame after dropping NaNs")
            continue

        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(df)

    except Exception as e:
        print(f"Skipping {run.name}: {e}")
        continue

# Plotting
plt.figure(figsize=(10, 6))

for group_key, dfs in grouped_data.items():
    df_concat = pd.concat(dfs, axis=0, ignore_index=True)
    df_grouped = df_concat.groupby("eval/eval_steps").agg(['mean', 'std'])

    steps = df_grouped.index
    mean = df_grouped[('eval/win_rate', 'mean')]
    std = df_grouped[('eval/win_rate', 'std')]

    plt.plot(steps, mean, label=f"{group_key} mean")
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

plt.xlabel("Evaluation Step")
plt.ylabel("Win Rate")
plt.title("Eval Win Rate by Run (Mean ± Std)")
plt.legend()
plt.tight_layout()

# Save to PDF
out_file_path = PLOT_DIR / "IL_vs_ILwithSS.pdf"

plt.savefig(out_file_path)
print("Done")
