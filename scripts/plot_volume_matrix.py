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

# Iterate through each unique label
for label in df['Label'].unique():
    # Filter dataframe for the current label
    df_label = df[df['Label'] == label]

    # Pivot the dataframe to create the matrix for the current label
    matrix_df = df_label.pivot("target", "source", "dt_Mbytes")

    # Calculate row and column averages
    row_averages = matrix_df.mean(axis=1).rename("Row Avg")
    column_averages = matrix_df.mean(axis=0).rename("Col Avg")

    # Append the averages to the dataframe
    matrix_df["Row Avg"] = row_averages
    column_averages_df = pd.DataFrame(column_averages).transpose()
    enhanced_matrix_df = pd.concat([matrix_df, column_averages_df], axis=0)

    # Plotting for each label
    plt.figure(figsize=(12, 10))
    sns.heatmap(enhanced_matrix_df, annot=True, fmt=".0f", cmap="Blues", cbar_kws={'label': 'Data Transfer (Mbytes)'})
    plt.title(f'Data Transfer Volume Matrix for {label}')
    plt.xlabel('Source Language')
    plt.ylabel('Target Language')
    plt.tight_layout()

    # Save the plot
    filename = f"./Volume Matrix {label}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}.")

    # Clear the figure to prevent overlap in plots
    plt.clf()

print("Done!")