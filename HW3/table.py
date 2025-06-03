import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
df_nosvd = pd.read_csv('HW3/results_summary_noSVD.csv')
df_svd = pd.read_csv('HW3/results_summary_SVD.csv')

# Add a column to indicate SVD usage
df_nosvd['SVD'] = 'No'
df_svd['SVD'] = 'Yes'

# Combine the two dataframes
combined_df = pd.concat([df_nosvd, df_svd])

# Create a new "Model + SVD" column for display purposes
combined_df['Model_SVD'] = combined_df['Model'] + ' (' + combined_df['SVD'] + ' SVD)'

# Reorder the columns
display_cols = ['Model_SVD', 'Train Acc', 'Val Acc', 'Test Acc', 'Train RMSE', 'Val RMSE', 'Test RMSE']
display_df = combined_df[display_cols].rename(columns={'Model_SVD': 'Model'})

# Create a visual table with results
results_data = [display_df.columns.tolist()]
results_data.extend(display_df.values.tolist())

# Plot the table as a figure
fig, ax = plt.subplots(figsize=(20, 8))
ax.axis('off')
ax.axis('tight')

# Create table
table = ax.table(cellText=results_data[1:], colLabels=results_data[0], 
                loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1,2)

# Highlight the best results in each category
best_test_acc_idx = combined_df['Test Acc'].idxmax()
best_test_rmse_idx = combined_df['Test RMSE'].idxmin()

for i, row in enumerate(results_data[1:]):
    # Highlight best Test Accuracy
    if i == best_test_acc_idx:
        table[(i+1), 3].set_facecolor('#C6EFCE')
    # Highlight best Test RMSE
    if i == best_test_rmse_idx:
        table[(i+1), 6].set_facecolor('#C6EFCE')

# Add a title
plt.title('Comparison of Models With and Without SVD', fontsize=16)
plt.tight_layout()
# Save as PNG
plt.savefig('HW3/combined_results_table.png', dpi=300, bbox_inches='tight')
plt.close()

print("Combined results table saved to HW3/combined_results_table.png")

# Also save the combined results to CSV
combined_df[['Model', 'SVD', 'Train Acc', 'Val Acc', 'Test Acc', 'Train RMSE', 'Val RMSE', 'Test RMSE']].to_csv(
    'HW3/combined_results_summary.csv', index=False)

print("Combined results also saved to HW3/combined_results_summary.csv")