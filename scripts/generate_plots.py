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

# Function to create plot with two metrics
def create_combined_plot(df, x_col, y_col1, y_col2, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    train_data = df[df[y_col1].notna()]
    plt.plot(train_data[x_col], train_data[y_col1], marker='o', label=y_col1)
    
    # Plot validation data
    val_data = df[df[y_col2].notna()]
    plt.plot(val_data[x_col], val_data[y_col2], marker='s', label=y_col2)
    
    plt.xlabel(x_col.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Plot created: {filename}")

# Create combined accuracy plot
create_combined_plot(df, 'step', 'train/acc', 'val/acc', 'Training and Validation Accuracy', 'Accuracy', 'train_acc.png')

# Create combined loss plot
create_combined_plot(df, 'step', 'train/loss', 'val/loss', 'Training and Validation Loss', 'Loss', 'train_loss.png')

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
