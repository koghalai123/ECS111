import pandas as pd
import os
import sys 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD



trainInfo = pd.read_csv('HW3/mv100k.train', sep='\t', header=None)
trainInfo.columns = [
    "user_id", "movie_id", "rating", "timestamp"
]

#movieInfo = pd.read_csv('HW3/info.item', sep='|', header=None)
movieColumns = pd.read_csv('HW3/genres.item', sep='|', header=None)

movieColumnsList1 = movieColumns[0].tolist()
movieColumnsList2 = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL']+movieColumnsList1


movieInfo = pd.read_csv('HW3/info.item', sep='|', header=None, names=movieColumnsList2, encoding='latin-1')
#print(movieInfo.head())


userInfo = pd.read_csv('HW3/info.user', sep='|', header=None)
userInfo.columns = [
    "user_id", "age", "gender", "occupation", "zip_code"
]

scaler = StandardScaler()
userInfo['age'] = scaler.fit_transform(userInfo[['age']])

combinedUserInfo = pd.merge(trainInfo[["user_id", "movie_id", "rating","timestamp"]], userInfo[["user_id", "age", "gender", "occupation"]], on='user_id', how='left')
combinedAllInfo = pd.merge(combinedUserInfo, movieInfo[['movie_id']+movieColumnsList1], on='movie_id', how='left')

encodedInfo = pd.get_dummies(combinedAllInfo, columns=['user_id','movie_id','gender', 'occupation'])


# Sort by timestamp
encodedSorted = encodedInfo.sort_values(by='timestamp')


split_idx = int(len(encodedSorted) * 0.8)

# First 80%
train = encodedSorted.iloc[:split_idx]
trainInput = train.drop(columns=["timestamp","rating"])
trainOutput = train["rating"]

# Last 20%
validate = encodedSorted.iloc[split_idx:]
validateInput = validate.drop(columns=["timestamp","rating"])
validateOutput = validate["rating"]

svd = TruncatedSVD(n_components=1500)  # Start with a large number
svd.fit(trainInput)
explained = np.cumsum(svd.explained_variance_ratio_)

plt.plot(explained)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid()
plt.show()

percentNeeded = 0.95
n_components = np.argmax(explained >= percentNeeded) + 1
print(f"Number of components for {percentNeeded} explanation: {n_components}")

# Fit SVD with the chosen number of components
svd_final = TruncatedSVD(n_components=n_components)
trainInput_svd = svd_final.fit_transform(trainInput)
validateInput_svd = svd_final.transform(validateInput)

print(f"Reduced from {train.shape[1]} to {n_components} components.")

# Convert SVD output to DataFrames, preserving index
svd_columns = [f'svd_{i}' for i in range(n_components)]
trainInput = pd.DataFrame(trainInput_svd, index=train.index, columns=svd_columns)
validateInput = pd.DataFrame(validateInput_svd, index=validate.index, columns=svd_columns)



logreg = LogisticRegression(penalty='l2', solver='liblinear',max_iter=300)
logreg.fit(trainInput, trainOutput)

trainPreds = logreg.predict(trainInput)
train_rmse_logreg = np.sqrt(np.mean((trainPreds - trainOutput) ** 2))
acc = accuracy_score(trainOutput, trainPreds)
print(f"Logistic Reg Train accuracy: {acc:.3f}, RMSE: {train_rmse_logreg:.3f}")

validatePreds = logreg.predict(validateInput)
validate_rmse_logreg = np.sqrt(np.mean((validatePreds - validateOutput) ** 2))
acc = accuracy_score(validateOutput, validatePreds)
print(f"Logistic Reg Validation accuracy: {acc:.3f}, RMSE: {validate_rmse_logreg:.3f}")



trainInput = trainInput.apply(pd.to_numeric)
validateInput = validateInput.apply(pd.to_numeric)

trainInput = trainInput.astype(float)
validateInput = validateInput.astype(float)

X_train = torch.tensor(trainInput.values, dtype=torch.float32)
X_val = torch.tensor(validateInput.values, dtype=torch.float32)

y_train = torch.tensor(trainOutput.values - 1, dtype=torch.long)
y_val = torch.tensor(validateOutput.values - 1, dtype=torch.long)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=100000, output_dim=5): 
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

input_dim = X_train.shape[1]
output_dim = len(torch.unique(y_train))  
model = SimpleNN(input_dim, hidden_dim=100, output_dim=output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

train_acc_list = []
val_acc_list = []

epochs = 2000
best_val_acc = 0
epochs_since_improvement = 0
patience = 100  

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Evaluation for this epoch
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).argmax(dim=1)
        train_acc = (train_preds == y_train).float().mean().item()
        # Convert predictions and targets back to original rating scale (add 1)
        train_preds_ratings = train_preds.cpu().numpy() + 1
        y_train_ratings = y_train.cpu().numpy() + 1
        train_rmse = np.sqrt(np.mean((train_preds_ratings - y_train_ratings) ** 2))

        val_preds = model(X_val).argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()
        val_preds_ratings = val_preds.cpu().numpy() + 1
        y_val_ratings = y_val.cpu().numpy() + 1
        val_rmse = np.sqrt(np.mean((val_preds_ratings - y_val_ratings) ** 2))

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        # Optionally, store RMSE for plotting
        if epoch == 0:
            train_rmse_list = []
            val_rmse_list = []
        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
              f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, "
              f"Train RMSE: {train_rmse:.3f}, Val RMSE: {val_rmse:.3f}")

    # Early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} (no val acc improvement for {patience} epochs)")
        break



# Evaluation
model.eval()
with torch.no_grad():
    train_preds = model(X_train).argmax(dim=1)
    train_acc = (train_preds == y_train).float().mean().item()
    val_preds = model(X_val).argmax(dim=1)
    val_acc = (val_preds == y_val).float().mean().item()

print(f"PyTorch NN Train accuracy: {train_acc:.3f}, RMSE: {train_rmse:.3f}")
print(f"PyTorch NN Validation accuracy: {val_acc:.3f}, RMSE: {val_rmse:.3f}")

# Optionally, plot RMSE curves
plt.figure(figsize=(10,5))
plt.plot(train_rmse_list, label='Train RMSE')
plt.plot(val_rmse_list, label='Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE')
plt.legend()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10,5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

print("done")


# Train a basic neural network (multi-layer perceptron)


# Training accuracy


# Validation accuracy












