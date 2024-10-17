import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Find the most recent metrics.csv file
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

# Function to create plot
def create_plot(df, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    valid_data = df[[x_col, y_col]].dropna()
    plt.plot(valid_data[x_col], valid_data[y_col], marker='o')
    plt.xlabel(x_col.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Plot created: {filename}")

# Create training loss plot
create_plot(df, 'step', 'train/loss', 'Training Loss over Steps', 'Loss', 'train_loss.png')

# Create training accuracy plot
create_plot(df, 'step', 'train/acc', 'Training Accuracy over Steps', 'Accuracy', 'train_acc.png')

# Generate test metrics table
test_metrics = df[df['test/acc'].notna()].iloc[-1]
test_table = "| Metric | Value |\n|--------|-------|\n"
test_table += f"| Test Accuracy | {test_metrics['test/acc']:.4f} |\n"
test_table += f"| Test Loss | {test_metrics['test/loss']:.4f} |\n"

# Write the test metrics table to a file
with open("test_metrics.md", "w") as f:
    f.write(test_table)
print("Test metrics table generated")

print("Script execution completed.")
