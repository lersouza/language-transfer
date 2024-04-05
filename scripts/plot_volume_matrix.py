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

# Display the first few rows of the dataframe to understand its structure
df.head()

# Convert dt from bytes to Mbytes
df['dt_Mbytes'] = df['dt'] / (1024 * 1024)

# Pivot the dataframe to create the matrix
matrix_df = df.pivot("target", "source", "dt_Mbytes")

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(matrix_df, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': 'Data Transfer (Mbytes)'})
plt.title('Data Transfer Volume Matrix')
plt.xlabel('Source Language')
plt.ylabel('Target Language')
plt.tight_layout()

# Show the plot
plt.show()
