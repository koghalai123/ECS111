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


# Perform logistic regression
model = LogisticRegression(max_iter=500, penalty='l2', solver='liblinear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Logistic Training Accuracy:", train_accuracy)
print("Logistic Testing Accuracy:", test_accuracy)


