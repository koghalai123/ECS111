import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_csv_data(csv_file):
    # Read the CSV file into a pandas DataFrame (assuming first row contains headers)
    try:
        data = pd.read_csv(csv_file)
        
        # Verify the required columns exist
        if not all(col in data.columns for col in ['x', 'y', 'label']):
            # Try to auto-detect columns if standard names aren't found
            if len(data.columns) >= 3:
                data.columns = ['x', 'y', 'label'] + list(data.columns[3:])
                print("Warning: Assuming first three columns are x, y, label")
            else:
                raise ValueError("CSV must contain at least three columns: x, y, label")
        
        # Get unique labels and assign a color to each
        unique_labels = data['label'].unique()
        num_labels = len(unique_labels)
        
        # Create a color map with enough distinct colors
        colors = cm.tab20(np.linspace(0, 1, num_labels)) if num_labels > 10 else cm.tab10(np.linspace(0, 1, num_labels))
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Plot each label with its assigned color
        for label, color in zip(unique_labels, colors):
            label_data = data[data['label'] == label]
            plt.scatter(label_data['x'], label_data['y'], 
                       color=color, label=label, alpha=0.7, edgecolors='w', s=50)
        
        # Add legend, title, and labels
        plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Scatter Plot of Data Points by Label')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Show the plot
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your CSV file has the correct format with headers: x,y,label")

# Example usage
if __name__ == "__main__":
    csv_file = "kmeans_results_test.csv"
    plot_csv_data(csv_file)