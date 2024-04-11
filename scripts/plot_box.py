import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default=Path("estimations/from_600M_10epochs.csv"))

args = parser.parse_args()

# Load the data
df = pd.read_csv(args.data_path)

# Convert 'dt' from bytes to MB and 'Pretraining size' to millions of tokens
df['dt_MB'] = df['dt'] / (1024 * 1024)
df['Pretraining size (Millions)'] = df['Pretraining size'] / 1e6

# Plotting
plt.figure(figsize=(14, 10))

# Extract unique pretraining sizes in millions of tokens
pretraining_sizes = sorted(df['Pretraining size (Millions)'].unique())

# Average values and sizes for the trend line
average_values = []
sizes_for_plot = []

for size in pretraining_sizes:
    data = df[df['Pretraining size (Millions)'] == size]['dt_MB']
    if not data.empty:
        plt.boxplot(data, positions=[size], widths=size/10, manage_ticks=False, showmeans=True)
        average_values.append(np.average(data))
        sizes_for_plot.append(size)

# Trend line
z = np.polyfit(sizes_for_plot, average_values, 1)
p = np.poly1d(z)
plt.plot(sizes_for_plot, p(sizes_for_plot), "r--", label=f'Slope: {z[0]:.5f}, Bias: {z[1]:.4f}')

plt.title('Data Transfer by Pretraining Size (Millions of Tokens vs. MB)')
plt.xlabel('Pretraining Size (Millions of Tokens)')
plt.ylabel('Data Transfer (MB)')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the figure
filename = "./Box Plot - Pretraining Size vs Data Transfer.png"
plt.savefig(filename)
plt.clf()  # Clear the figure after saving
print(f"Box plot saved as {filename}")
