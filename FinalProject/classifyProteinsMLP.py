import torch
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import csv

# Load the combined DataFrame from CSV
combined_df = pd.read_csv("FinalProject/combined_proteins.csv")

# Prepare data
X = combined_df.drop(columns=['label'])
y = combined_df['label'].astype(int)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, random_state=42, stratify=y_trainval)

# Convert to torch tensors
def to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

X_train = to_tensor(X_train)
X_val = to_tensor(X_val)
X_test = to_tensor(X_test)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val = torch.tensor(y_val.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# Define a flexible MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# --- Hyperparameter sweeps ---
input_dim = X_train.shape[1]
output_dim = len(torch.unique(y_train))

# 1. Vary number of neurons per layer (single hidden layer)
neurons_list = [16, 32, 64, 128, 256,512,1024,2048,4096,8192,16384]
epochs = 50
best_val_acc_neurons = 0
best_neurons = None
train_acc_neurons, val_acc_neurons = [], []

for n in neurons_list:
    model = MLP(input_dim, [n], output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean().item()
        val_acc = (model(X_val).argmax(dim=1) == y_val).float().mean().item()
    train_acc_neurons.append(train_acc)
    val_acc_neurons.append(val_acc)
    if val_acc > best_val_acc_neurons:
        best_val_acc_neurons = val_acc
        best_neurons = n
    print(f"[MLP neurons sweep] Neurons: {n}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

plt.figure()
plt.plot(neurons_list, val_acc_neurons, marker='o', label='Validation')
plt.plot(neurons_list, train_acc_neurons, marker='x', linestyle='--', label='Train')
plt.xlabel('Neurons in Hidden Layer')
plt.ylabel('Accuracy')
plt.title('MLP Accuracy vs. Neurons per Layer')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_accuracy_vs_neurons.png")

# 2. Vary number of hidden layers (fixed neurons per layer)
layers_list = [1, 2, 4, 8, 16 ,32]
neurons_per_layer = 64
epochs = 50
best_val_acc_layers = 0
best_layers = None
train_acc_layers, val_acc_layers = [], []

for num_layers in layers_list:
    model = MLP(input_dim, [neurons_per_layer]*num_layers, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean().item()
        val_acc = (model(X_val).argmax(dim=1) == y_val).float().mean().item()
    train_acc_layers.append(train_acc)
    val_acc_layers.append(val_acc)
    if val_acc > best_val_acc_layers:
        best_val_acc_layers = val_acc
        best_layers = num_layers
    print(f"[MLP layers sweep] Layers: {num_layers}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

plt.figure()
plt.plot(layers_list, val_acc_layers, marker='o', label='Validation')
plt.plot(layers_list, train_acc_layers, marker='x', linestyle='--', label='Train')
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy')
plt.title('MLP Accuracy vs. Number of Layers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_accuracy_vs_layers.png")

# 3. Vary number of epochs (fixed architecture)
epochs_list = [10, 20, 50, 100, 200,500,1000,10000]
neurons = 64
layers = 2
best_val_acc_epochs = 0
best_epochs = None
train_acc_epochs, val_acc_epochs = [], []

for num_epochs in epochs_list:
    model = MLP(input_dim, [neurons]*layers, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean().item()
        val_acc = (model(X_val).argmax(dim=1) == y_val).float().mean().item()
    train_acc_epochs.append(train_acc)
    val_acc_epochs.append(val_acc)
    if val_acc > best_val_acc_epochs:
        best_val_acc_epochs = val_acc
        best_epochs = num_epochs
    print(f"[MLP epochs sweep] Epochs: {num_epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

plt.figure()
plt.plot(epochs_list, val_acc_epochs, marker='o', label='Validation')
plt.plot(epochs_list, train_acc_epochs, marker='x', linestyle='--', label='Train')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('MLP Accuracy vs. Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_accuracy_vs_epochs.png")

plt.close('all')

# --- Evaluate on test set with best hyperparameters ---
# Best neurons
model_best_neurons = MLP(input_dim, [best_neurons], output_dim).to(device)
optimizer = optim.Adam(model_best_neurons.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(50):
    model_best_neurons.train()
    optimizer.zero_grad()
    loss = criterion(model_best_neurons(X_train), y_train)
    loss.backward()
    optimizer.step()
model_best_neurons.eval()
with torch.no_grad():
    test_acc_best_neurons = (model_best_neurons(X_test).argmax(dim=1) == y_test).float().mean().item()

# Best layers
model_best_layers = MLP(input_dim, [neurons_per_layer]*best_layers, output_dim).to(device)
optimizer = optim.Adam(model_best_layers.parameters(), lr=0.001)
for epoch in range(50):
    model_best_layers.train()
    optimizer.zero_grad()
    loss = criterion(model_best_layers(X_train), y_train)
    loss.backward()
    optimizer.step()
model_best_layers.eval()
with torch.no_grad():
    test_acc_best_layers = (model_best_layers(X_test).argmax(dim=1) == y_test).float().mean().item()

# Best epochs
model_best_epochs = MLP(input_dim, [neurons]*layers, output_dim).to(device)
optimizer = optim.Adam(model_best_epochs.parameters(), lr=0.001)
for epoch in range(best_epochs):
    model_best_epochs.train()
    optimizer.zero_grad()
    loss = criterion(model_best_epochs(X_train), y_train)
    loss.backward()
    optimizer.step()
model_best_epochs.eval()
with torch.no_grad():
    test_acc_best_epochs = (model_best_epochs(X_test).argmax(dim=1) == y_test).float().mean().item()

# --- Combined optimal hyperparameters case ---
# Use the best values found from the individual sweeps
opt_neurons = int(best_neurons)
opt_layers = int(best_layers)
opt_epochs = int(best_epochs)
model_opt_all = MLP(input_dim, [opt_neurons]*opt_layers, output_dim).to(device)
optimizer = optim.Adam(model_opt_all.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


train_acc_opt_all = []
val_acc_opt_all = []

for epoch in range(opt_epochs):
    model_opt_all.train()
    optimizer.zero_grad()
    loss = criterion(model_opt_all(X_train), y_train)
    loss.backward()
    optimizer.step()
    # Report accuracy after each epoch
    model_opt_all.eval()
    with torch.no_grad():
        train_acc = (model_opt_all(X_train).argmax(dim=1) == y_train).float().mean().item()
        val_acc = (model_opt_all(X_val).argmax(dim=1) == y_val).float().mean().item()
    train_acc_opt_all.append(train_acc)
    val_acc_opt_all.append(val_acc)
    print(f"[MLP optimal-all sweep][Epoch {epoch+1}/{opt_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

model_opt_all.eval()
with torch.no_grad():
    test_acc_opt_all = (model_opt_all(X_test).argmax(dim=1) == y_test).float().mean().item()
print(f"[MLP optimal-all sweep] Neurons: {opt_neurons}, Layers: {opt_layers}, Epochs: {opt_epochs}, Test Acc: {test_acc_opt_all:.4f}")

# Optionally, plot the learning curve for the optimal-all case
plt.figure()
plt.plot(range(1, opt_epochs+1), train_acc_opt_all, label='Train Accuracy')
plt.plot(range(1, opt_epochs+1), val_acc_opt_all, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MLP Train/Validation Accuracy (Optimal Hyperparameters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_optimal_all_learning_curve.png")

# --- Save best test accuracies to CSV ---
with open("mlp_best_test_accuracies.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Experiment", "Best Hyperparameter", "Test Accuracy"])
    writer.writerow(["neurons sweep", f"neurons={best_neurons}", f"{test_acc_best_neurons:.4f}"])
    writer.writerow(["layers sweep", f"layers={best_layers}", f"{test_acc_best_layers:.4f}"])
    writer.writerow(["epochs sweep", f"epochs={best_epochs}", f"{test_acc_best_epochs:.4f}"])
    writer.writerow(
        ["optimal-all sweep", f"neurons={opt_neurons}, layers={opt_layers}, epochs={opt_epochs}", f"{test_acc_opt_all:.4f}"]
    )

