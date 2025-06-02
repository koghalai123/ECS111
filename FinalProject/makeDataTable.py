import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Directory containing the CSV files
csv_dir = "FinalProject/Plots"

# Find all *_best_test_accuracies.csv files
csv_files = glob.glob(os.path.join(csv_dir, "*_best_test_accuracies.csv"))

# Collect all results
results = []

for csv_file in csv_files:
    # Determine algorithm from filename
    if "svm" in csv_file.lower():
        algo = "SVM"
    elif "logreg" in csv_file.lower():
        algo = "Logistic Regression"
    elif "mlp" in csv_file.lower():
        algo = "MLP"
    else:
        algo = "Unknown"
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        results.append({
            "Algorithm": algo,
            "Experiment": row["Experiment"],
            "Hyperparameter": row["Best Hyperparameter"],
            "Test Accuracy": float(row["Test Accuracy"])
        })

# Create a DataFrame for display
results_df = pd.DataFrame(results)
results_df = results_df[["Algorithm", "Experiment", "Hyperparameter", "Test Accuracy"]]
# Sort by Algorithm first, then by Experiment
results_df = results_df.sort_values(by=['Algorithm', 'Experiment'])

# Normalize accuracy for color mapping
accs = results_df["Test Accuracy"].astype(float)
norm = plt.Normalize(accs.min(), accs.max())
cmap = plt.get_cmap("YlGn")

# Plot the table as a figure and save as PNG with higher resolution and larger size
fig, ax = plt.subplots(figsize=(22, 2 + 2*len(results_df)))
ax.axis('off')
table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(24)
table.auto_set_column_width(col=list(range(len(results_df.columns))))
table.scale(1, 4)

# Shade the "Test Accuracy" column based on value, but blend with white for lighter colors
for i in range(len(results_df)):
    acc = float(results_df.iloc[i]["Test Accuracy"])
    color = cmap(norm(acc))
    # Blend with white for lighter background (alpha controls how much white is mixed in)
    alpha = 0.4  # Increase for lighter color (0.0 = original, 1.0 = white)
    white = np.array([1, 1, 1, 1])
    color = tuple((1 - alpha) * np.array(color) + alpha * white)
    table[i+1, 3].set_facecolor(color)  # +1 for header row

plt.tight_layout()
plt.savefig("FinalProject/Plots/classifier_test_accuracies_table.png", dpi=300)
plt.close()