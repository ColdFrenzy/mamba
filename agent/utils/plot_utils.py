
from agent.utils.paths import PLOT_DIR
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import matplotlib

matplotlib.use('Agg')

def extract_run_name_without_seed(col_name):
    """Extracts base run name without seed and metric suffix."""
    # Remove the prefix "Name: " if it exists
    name = col_name.replace("Name: ", "")
    # Remove suffix after ' - '
    name = name.split(" - ")[0]
    # Remove SEED=... pattern
    name = re.sub(r"_SEED=\d+", "", name)
    return name

def clean_and_convert(series):
    """Strips and converts to numeric, safely."""
    return pd.to_numeric(series.astype(str).str.strip(), errors='coerce')

def plot_grouped_win_rate(csv_path, output_pdf_path="win_rate_grouped.pdf", legend_name_map=None):
    df = pd.read_csv(csv_path)

    # Get all step and win_rate columns
    steps_col = df["eval/eval_steps"].name
    win_cols = [col for col in df.columns if "eval/win_rate" in col and "MIN" not in col and "MAX" not in col]

    # Group columns by cleaned name (excluding seed info)
    grouped = {}

    for win_col in win_cols:
        group_name = extract_run_name_without_seed(win_col)
        if group_name not in grouped:
            grouped[group_name] = []
        grouped[group_name].append((steps_col, win_col))

    # Plot
    plt.figure(figsize=(10, 6))

    for group_name, col_pairs in grouped.items():
        all_steps = []
        all_win_rates = []

        for step_col, win_col in col_pairs:
            steps = clean_and_convert(df[step_col])
            win_rates = clean_and_convert(df[win_col])
            valid_mask = steps.notna() & win_rates.notna()
            all_steps.append(steps[valid_mask].values)
            all_win_rates.append(win_rates[valid_mask].values)

        # Stack and interpolate missing steps across seeds
        # Assume all seeds share the same steps
        if len(all_steps) > 0:
            steps_common = all_steps[0]
            win_matrix = pd.DataFrame(all_win_rates).T  # shape: [timesteps, seeds]
            win_mean = win_matrix.mean(axis=1)
            win_std = win_matrix.std(axis=1)

            label = legend_name_map.get(group_name, group_name)  # fallback to original name
            plt.plot(steps_common, win_mean, label=label)
            plt.fill_between(steps_common, win_mean - win_std, win_mean + win_std, alpha=0.2)


    plt.xlabel("Steps")
    plt.ylabel("Win Rate")
    plt.title("Win Rate (mean ± std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    plt.savefig(output_pdf_path)
    print(f"✅ Grouped plot saved to: {os.path.abspath(output_pdf_path)}")



if __name__ == "__main__":
    in_file_path = PLOT_DIR / "IL_vs_ILwithSS.csv"
    out_file_path = PLOT_DIR / "IL_vs_ILwithSS.pdf"
    legend_name_map = {
        "MultiDreamer_3m_SR_AC_LSV_100.0K": "Independent Dreamer + Strategy Selector",
        "MultiDreamer_3m_SR_AC_100.0K": "Independent Dreamer",
        # Add more mappings as needed
    }
    
    plot_grouped_win_rate(in_file_path, out_file_path, legend_name_map)
