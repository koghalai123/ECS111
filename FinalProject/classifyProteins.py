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

'''# SVM Model
model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
trainAccuracy = accuracy_score(y_train, y_pred)
print("SVM Accuracy Train:", trainAccuracy)
y_pred = model.predict(X_test)
testAccuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy Test:", testAccuracy)



# Perform logistic regression
model = LogisticRegression(max_iter=500, penalty='l2', solver='liblinear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Logistic Training Accuracy:", train_accuracy)
print("Logistic Testing Accuracy:", test_accuracy)'''





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

'''for epoch in range(epochs):
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
plt.show()'''



# Define the convolutional neural network
class ProteinClassifierCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ProteinClassifierCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * (input_dim // 4), 128)  # Adjust input_dim based on pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Initialize the CNN model
input_dim = X_train.shape[1]
model_cnn = ProteinClassifierCNN(input_dim=input_dim, num_classes=len(torch.unique(y_train)))

# Loss function and optimizer
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

# Move data and model to device
model_cnn.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# Create DataLoader for mini-batches
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 64  # Reduce batch size to fit in GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop with DataLoader
train_acc_list_cnn = []
val_acc_list_cnn = []
train_loss_list_cnn = []
val_loss_list_cnn = []

epochs = 100
patience = 10
best_val_loss_cnn = float('inf')
epochs_since_improvement_cnn = 0

for epoch in range(epochs):
    model_cnn.train()
    train_loss = 0.0
    train_correct = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer_cnn.zero_grad()
        outputs_cnn = model_cnn(batch_X)
        loss_cnn = criterion_cnn(outputs_cnn, batch_y)
        loss_cnn.backward()
        optimizer_cnn.step()
        
        train_loss += loss_cnn.item()
        train_preds = outputs_cnn.argmax(dim=1)
        train_correct += (train_preds == batch_y).sum().item()
    
    train_acc_cnn = train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader)

    # Validation loop
    model_cnn.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            val_outputs_cnn = model_cnn(batch_X)
            val_loss += criterion_cnn(val_outputs_cnn, batch_y).item()
            val_preds = val_outputs_cnn.argmax(dim=1)
            val_correct += (val_preds == batch_y).sum().item()
    
    val_acc_cnn = val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc_cnn:.3f}, Val Acc: {val_acc_cnn:.3f}")
    
    # Early stopping
    if val_loss < best_val_loss_cnn:
        best_val_loss_cnn = val_loss
        epochs_since_improvement_cnn = 0
    else:
        epochs_since_improvement_cnn += 1
    
    if epochs_since_improvement_cnn >= patience:
        print(f"Early stopping at epoch {epoch+1} (no val loss improvement for {patience} epochs)")
        break

# Final evaluation for CNN using mini-batches
model_cnn.eval()
train_correct = 0
val_correct = 0

# Evaluate on training data
with torch.no_grad():
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        train_outputs_cnn = model_cnn(batch_X)
        train_preds_cnn = train_outputs_cnn.argmax(dim=1)
        train_correct += (train_preds_cnn == batch_y).sum().item()

train_acc_cnn = train_correct / len(train_loader.dataset)

# Evaluate on validation data
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        val_outputs_cnn = model_cnn(batch_X)
        val_preds_cnn = val_outputs_cnn.argmax(dim=1)
        val_correct += (val_preds_cnn == batch_y).sum().item()

val_acc_cnn = val_correct / len(val_loader.dataset)

print(f"Final CNN Train Accuracy: {train_acc_cnn:.3f}")
print(f"Final CNN Validation Accuracy: {val_acc_cnn:.3f}")

# Plot accuracy curves for CNN
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list_cnn, label='Train Accuracy (CNN)')
plt.plot(val_acc_list_cnn, label='Validation Accuracy (CNN)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (CNN)')
plt.legend()
plt.show()

# Plot loss curves for CNN
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list_cnn, label='Train Loss (CNN)')
plt.plot(val_loss_list_cnn, label='Validation Loss (CNN)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (CNN)')
plt.legend()
plt.show()


print('done')