import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default=Path("estimations/from_600M_10epochs.csv"))

args = parser.parse_args()

# Load the data
df = pd.read_csv(args.data_path)

# Convert dt from bytes to Mbytes
df['dt_Mbytes'] = df['dt'] / (1024 * 1024)

# Pivot the dataframe to create the matrix
matrix_df = df.pivot("target", "source", "dt_Mbytes")

# Calculate row and column averages
row_averages = matrix_df.mean(axis=1).rename("Row Avg")
column_averages = matrix_df.mean(axis=0).rename("Col Avg")

# Append the averages to the dataframe
matrix_df["Row Avg"] = row_averages
column_averages_df = pd.DataFrame(column_averages).transpose()
enhanced_matrix_df = pd.concat([matrix_df, column_averages_df], axis=0)

# Plotting
plt.figure(figsize=(12, 10))
sns.heatmap(enhanced_matrix_df, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': 'Data Transfer (Mbytes)'})
plt.title(f'Data Transfer Volume Matrix {args.data_path}')
plt.xlabel('Source Language')
plt.ylabel('Target Language')
plt.tight_layout()

# Show the plot
plt.show()
