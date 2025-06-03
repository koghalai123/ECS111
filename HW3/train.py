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
import copy


#read in movie watch data
trainInfo = pd.read_csv('HW3/mv100k.train', sep='\t', header=None)
trainInfo.columns = [
    "user_id", "movie_id", "rating", "timestamp"
]

#movieInfo = pd.read_csv('HW3/info.item', sep='|', header=None)
movieColumns = pd.read_csv('HW3/genres.item', sep='|', header=None)
movieColumnsList1 = movieColumns[0].tolist()
movieColumnsList2 = ['movie_id','movie_title','release_date', 'video_release_date','IMDb_URL']+movieColumnsList1

#read in movie info data
movieInfo = pd.read_csv('HW3/info.item', sep='|', header=None, names=movieColumnsList2, encoding='latin-1')
#print(movieInfo.head())

#read in user info data
userInfo = pd.read_csv('HW3/info.user', sep='|', header=None)
userInfo.columns = [
    "user_id", "age", "gender", "occupation", "zip_code"
]

#normalize age
scaler = StandardScaler()
userInfo['age'] = scaler.fit_transform(userInfo[['age']])

#combine user and movie data
combinedUserInfo = pd.merge(trainInfo[["user_id", "movie_id", "rating","timestamp"]], userInfo[["user_id", "age", "gender", "occupation"]], on='user_id', how='left')
combinedAllInfo = pd.merge(combinedUserInfo, movieInfo[['movie_id']+movieColumnsList1], on='movie_id', how='left')
#one hot encode the data
encodedInfo = pd.get_dummies(combinedAllInfo, columns=['user_id','movie_id','gender', 'occupation'])


# Sort by timestamp then split into training and validation sets
encodedSorted = encodedInfo.sort_values(by='timestamp')
split_idx = int(len(encodedSorted) * 0.8)

train = encodedSorted.iloc[:split_idx]
trainInput = train.drop(columns=["timestamp","rating"])
trainOutput = train["rating"]

validate = encodedSorted.iloc[split_idx:]
validateInput = validate.drop(columns=["timestamp","rating"])
validateOutput = validate["rating"]


#implement SVD. This can be commentede out if not needed
svd = TruncatedSVD(n_components=1500)  # Start with a large number
svd.fit(trainInput)
explained = np.cumsum(svd.explained_variance_ratio_)

#plot of SVD feature explained variance
plt.plot(explained)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid()
#plt.show()
plt.savefig('HW3/SVDPlot.png', dpi=300)
plt.close()

# Fit SVD with the chosen number of components
percentNeeded = 0.95
n_components = np.argmax(explained >= percentNeeded) + 1
print(f"Number of components for {percentNeeded} explanation: {n_components}")
svd_final = TruncatedSVD(n_components=n_components)
trainInput_svd = svd_final.fit_transform(trainInput)
validateInput_svd = svd_final.transform(validateInput)
print(f"Reduced from {train.shape[1]} to {n_components} components.")

# Convert SVD output to be usable later
svd_columns = [f'svd_{i}' for i in range(n_components)]
trainInput = pd.DataFrame(trainInput_svd, index=train.index, columns=svd_columns)
validateInput = pd.DataFrame(validateInput_svd, index=validate.index, columns=svd_columns)


#implement logistic regression
logreg = LogisticRegression(penalty='l2', solver='liblinear',max_iter=300)
logreg.fit(trainInput, trainOutput)

#train on training data
trainPreds = logreg.predict(trainInput)
train_rmse_logreg = np.sqrt(np.mean((trainPreds - trainOutput) ** 2))
acc = accuracy_score(trainOutput, trainPreds)
print(f"Logistic Reg Train accuracy: {acc:.3f}, RMSE: {train_rmse_logreg:.3f}")
#validate on validation data
validatePreds = logreg.predict(validateInput)
validate_rmse_logreg = np.sqrt(np.mean((validatePreds - validateOutput) ** 2))
acc = accuracy_score(validateOutput, validatePreds)
print(f"Logistic Reg Validation accuracy: {acc:.3f}, RMSE: {validate_rmse_logreg:.3f}")


#prepare data to be used in MLP
trainInput = trainInput.apply(pd.to_numeric)
validateInput = validateInput.apply(pd.to_numeric)

trainInput = trainInput.astype(float)
validateInput = validateInput.astype(float)

X_train = torch.tensor(trainInput.values, dtype=torch.float32)
X_val = torch.tensor(validateInput.values, dtype=torch.float32)

y_train = torch.tensor(trainOutput.values - 1, dtype=torch.long)
y_val = torch.tensor(validateOutput.values - 1, dtype=torch.long)

#MLP class
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

#prepare model
input_dim = X_train.shape[1]
output_dim = len(torch.unique(y_train))  
model = SimpleNN(input_dim, hidden_dim=500, output_dim=output_dim)

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

#train model
epochs = 2000
best_val_acc = 0
best_model_state = None
epochs_since_improvement = 0
patience = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Evaluation for each epoch
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

        # store RMSE for plotting
        if epoch == 0:
            train_rmse_list = []
            val_rmse_list = []
        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

        # Save model if it's best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
            print(f"New best model saved at epoch {epoch+1} with val accuracy: {val_acc:.3f}")
        else:
            epochs_since_improvement += 1
    #print progress
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
              f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, "
              f"Train RMSE: {train_rmse:.3f}, Val RMSE: {val_rmse:.3f}")
    #allow for early stopping
    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1} (no val acc improvement for {patience} epochs)")
        break

print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")

# Load the best model for evaluation
print("Loading best model for evaluation...")
model.load_state_dict(best_model_state)


# Evaluation
model.eval()
with torch.no_grad():
    train_preds = model(X_train).argmax(dim=1)
    train_acc = (train_preds == y_train).float().mean().item()
    val_preds = model(X_val).argmax(dim=1)
    val_acc = (val_preds == y_val).float().mean().item()

print(f"PyTorch NN Train accuracy: {train_acc:.3f}, RMSE: {train_rmse:.3f}")
print(f"PyTorch NN Validation accuracy: {val_acc:.3f}, RMSE: {val_rmse:.3f}")

# plot RMSE curves
plt.figure(figsize=(10,5))
plt.plot(train_rmse_list, label='Train RMSE')
plt.plot(val_rmse_list, label='Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE')
plt.legend()
plt.savefig('HW3/rmse_curves.png', dpi=300)
plt.close()

# Plot accuracy curves
plt.figure(figsize=(10,5))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('HW3/accuracy_curves.png', dpi=300)
plt.close()


# Load and process test data
testInfo = pd.read_csv('HW3/mv100k.test', sep='\t', header=None)
testInfo.columns = ["user_id", "movie_id", "rating", "timestamp"]

# Process test data the same way as training data
testCombinedUserInfo = pd.merge(testInfo[["user_id", "movie_id", "rating","timestamp"]], 
                               userInfo[["user_id", "age", "gender", "occupation"]], 
                               on='user_id', how='left')
testCombinedAllInfo = pd.merge(testCombinedUserInfo, 
                              movieInfo[['movie_id']+movieColumnsList1], 
                              on='movie_id', how='left')
testEncoded = pd.get_dummies(testCombinedAllInfo, columns=['user_id','movie_id','gender', 'occupation'])

# Handle missing columns 
train_cols = set(encodedInfo.columns)
test_cols = set(testEncoded.columns)

missing_cols = train_cols - test_cols
if missing_cols:
    missing_df = pd.DataFrame(0, index=testEncoded.index, columns=list(missing_cols))
    testEncoded = pd.concat([testEncoded, missing_df], axis=1)
    testEncoded = testEncoded.copy()

# Process test data similarly to how we processed the training data
testEncodedInput = testEncoded.drop(columns=["timestamp", "rating"])
testOutput = testEncoded["rating"]

# Check if SVD is being used 
if 'svd_0' in trainInput.columns:
    print("Applying SVD transformation to test data...")
    
    # Handle columns that appear in test but not training data
    train_feature_cols = trainInput.columns if 'svd_0' not in trainInput.columns else encodedInfo.drop(columns=["timestamp", "rating"]).columns
    common_cols = [col for col in testEncodedInput.columns if col in train_feature_cols]
    missing_cols = [col for col in train_feature_cols if col not in testEncodedInput.columns]
    
    for col in missing_cols:
        testEncodedInput[col] = 0
        
    testEncodedInput_filtered = testEncodedInput[train_feature_cols]
    
    # Apply SVD transformation
    testInput_svd = svd_final.transform(testEncodedInput_filtered)
    
    # Convert to dataframe
    testInput = pd.DataFrame(testInput_svd, index=testEncodedInput.index, 
                            columns=[f'svd_{i}' for i in range(n_components)])
else:
    testInput = testEncodedInput[[col for col in trainInput.columns if col in testEncodedInput.columns]]
    
# Ensure data types match training data
testInput = testInput.apply(pd.to_numeric)
testInput = testInput.astype(float)

# Test logistic regression
test_preds_logreg = logreg.predict(testInput)
test_rmse_logreg = np.sqrt(np.mean((test_preds_logreg - testOutput) ** 2))
test_acc_logreg = accuracy_score(testOutput, test_preds_logreg)
print(f"Logistic Regression Test accuracy: {test_acc_logreg:.3f}, RMSE: {test_rmse_logreg:.3f}")

# Test MLP
X_test = torch.tensor(testInput.values, dtype=torch.float32).to(device)
y_test = torch.tensor(testOutput.values - 1, dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    test_preds = model(X_test).argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean().item()
    test_preds_ratings = test_preds.cpu().numpy() + 1
    y_test_ratings = y_test.cpu().numpy() + 1
    test_rmse = np.sqrt(np.mean((test_preds_ratings - y_test_ratings) ** 2))
    
print(f"PyTorch NN (Best Model) Test accuracy: {test_acc:.3f}, RMSE: {test_rmse:.3f}")

# Create a table
results_data = [
    ['Model', 'Train Acc', 'Val Acc', 'Test Acc', 'Train RMSE', 'Val RMSE', 'Test RMSE'],
    ['Logistic Regression', f"{accuracy_score(trainOutput, trainPreds):.3f}", 
     f"{accuracy_score(validateOutput, validatePreds):.3f}", f"{test_acc_logreg:.3f}", 
     f"{train_rmse_logreg:.3f}", f"{validate_rmse_logreg:.3f}", f"{test_rmse_logreg:.3f}"],
    ['Neural Network', f"{train_acc:.3f}", f"{val_acc:.3f}", f"{test_acc:.3f}", 
     f"{train_rmse:.3f}", f"{val_rmse:.3f}", f"{test_rmse:.3f}"]
]

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=results_data[1:], colLabels=results_data[0], 
                loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)  

plt.tight_layout()
plt.savefig('HW3/results_table.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results to CSV
results_df = pd.DataFrame(results_data[1:], columns=results_data[0])
results_df.to_csv('HW3/results_summary.csv', index=False)

print("Results saved")
print("done")



