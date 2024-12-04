import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_path = "~/Downloads/Leandro - Data Transfer - D_T All.csv"  # Adjust the path to your file
df = pd.read_csv(data_path)

# Convert 'dt' from bytes to MB
df['dt_MB'] = df['dt'] / (1024 * 1024)

# Filter data for the two specific labels
data_600M_10epochs = df[df['Label'] == '600M 10 epochs']['dt_MB']
data_6B_1epoch = df[df['Label'] == '6B 1 epoch']['dt_MB']

# Prepare data for plotting
data_to_plot = [data_600M_10epochs, data_6B_1epoch]
labels = ['600M 10 epochs', '6B 1 epoch']

# Plotting
plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=labels)
plt.title('Data Transfer Comparison')
plt.ylabel('Data Transfer (MB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
