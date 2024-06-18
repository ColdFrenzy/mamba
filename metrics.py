import numpy as np

# Dictionary to store win rates for each algorithm
algorithms_win_rates = {
    'MAMBA_no_ac_4_strat_strat_adv_traj_synth': [17, 30, 15],
    # 'MAMBA_no_ac': [9, 27, 36]
    # Add more algorithms as needed
}

# Calculate mean and std for each algorithm, rounding to integers
results = {}
for algo, win_rates in algorithms_win_rates.items():
    mean_percentage = round(np.mean(win_rates))
    std_percentage = round(np.std(win_rates))
    results[algo] = {
        'mean_percentage': mean_percentage,
        'std_percentage': std_percentage
    }

# Print results
for algo, metrics in results.items():
    print(f"{algo}: Mean Win Rate = {metrics['mean_percentage']}%, Standard Deviation = {metrics['std_percentage']}%")
