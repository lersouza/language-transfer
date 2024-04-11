import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_path = "~/Downloads/Leandro - Data Transfer - D_T All.csv"  # Adjust the path to your file
df = pd.read_csv(data_path)

# Convert 'dt' from bytes to MB
df['dt_MB'] = df['dt'] / (1024 * 1024)

# Filter data for the two specific labels
data_6M_10epochs = df[df['Label'] == '6B 3 epochs']['dt_MB']
data_60M_1epoch = df[df['Label'] == '6B 3 epochs, ft 60M 1 epoch']['dt_MB']

# Prepare data for plotting
data_to_plot = [data_6M_10epochs, data_60M_1epoch]
labels = ['6M 10 epochs', '60M 1 epoch']

# Plotting
plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=labels)
plt.title('Data Transfer Comparison')
plt.ylabel('Data Transfer (MB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
