import pickle
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC


import pandas as pd
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD


'''# Load the pickle files
with open("FinalProject/low_embeddings.pkl", "rb") as file:  
    low_temp_proteins = pickle.load(file)
    
with open("FinalProject/med_embeddings.pkl", "rb") as file:  
    med_temp_proteins = pickle.load(file)

# Convert pickle objects to pandas DataFrames
def convert_to_dataframe(proteins, label_value):
    data = []
    for entry in proteins:
        mean_representations = entry['mean_representations']
        flattened_representation = {f"feature_{key}_{i}": value.numpy().flatten()[i] 
                                    for key, value in mean_representations.items() 
                                    for i in range(value.numpy().flatten().shape[0])}
        flattened_representation['label'] = label_value  # Ensure label is discrete (e.g., 0 or 1)
        data.append(flattened_representation)
    return pd.DataFrame(data)

# Convert both pickle objects to DataFrames
low_df = convert_to_dataframe(low_temp_proteins, label_value=0)  # Label 0 for low temperature proteins
med_df = convert_to_dataframe(med_temp_proteins, label_value=1)  # Label 1 for medium temperature proteins

# Concatenate the DataFrames
combined_df = pd.concat([low_df, med_df], ignore_index=True)
combined_df.to_csv("combined_proteins.csv", index=False)
print("Combined DataFrame saved to combined_proteins.csv")'''

# Load the combined DataFrame from CSV
combined_df = pd.read_csv("FinalProject/combined_proteins.csv")

# Prepare data for logistic regression
X = combined_df.drop(columns=['label'])  # Features
y = combined_df['label'].astype(int)  # Ensure labels are integers

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocess the data
trainInput = X_train.apply(pd.to_numeric)
validateInput = X_test.apply(pd.to_numeric)

trainInput = trainInput.astype(float)
validateInput = validateInput.astype(float)

X_train = torch.tensor(trainInput.values, dtype=torch.float32)
X_val = torch.tensor(validateInput.values, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)  # Ensure labels are integers
y_val = torch.tensor(y_test.values, dtype=torch.long)

# Define the neural network
class ProteinClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):  # Binary classification
        super(ProteinClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_dim = X_train.shape[1]
output_dim = len(torch.unique(y_train))  # Number of classes
model = ProteinClassifierNN(input_dim, hidden_dim=128, output_dim=output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move data and model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# Training loop
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

epochs = 100
patience = 10
best_val_loss = float('inf')
epochs_since_improvement = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Evaluate on training data
    train_preds = model(X_train).argmax(dim=1)
    train_acc = (train_preds == y_train).float().mean().item()
    train_loss = loss.item()
    
    # Evaluate on validation data
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()
    
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1
    
    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} (no val loss improvement for {patience} epochs)")
        break

# Final evaluation
model.eval()
with torch.no_grad():
    train_preds = model(X_train).argmax(dim=1)
    train_acc = (train_preds == y_train).float().mean().item()
    val_preds = model(X_val).argmax(dim=1)
    val_acc = (val_preds == y_val).float().mean().item()

print(f"Final Train Accuracy: {train_acc:.3f}")
print(f"Final Validation Accuracy: {val_acc:.3f}")

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

