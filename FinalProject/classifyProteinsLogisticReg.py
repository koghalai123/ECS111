import pickle
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load the combined DataFrame from CSV
combined_df = pd.read_csv("FinalProject/combined_proteins.csv")

# Prepare data for logistic regression
X = combined_df.drop(columns=['label'])  # Features
y = combined_df['label'].astype(int)  # Ensure labels are integers

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, random_state=42, stratify=y_trainval)

# --- Vary C, keep max_iter fixed ---
C_values = [0.01, 0.1, 1, 10, 100]
train_acc_C = []
val_acc_C = []
for C in C_values:
    model = LogisticRegression(max_iter=500, penalty='l2', solver='liblinear', C=C)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    train_acc_C.append(accuracy_score(y_train, y_pred_train))
    val_acc_C.append(accuracy_score(y_val, y_pred_val))
    print(f"[C sweep] C: {C}, Train Acc: {train_acc_C[-1]:.4f}, Val Acc: {val_acc_C[-1]:.4f}")

# --- Vary max_iter, keep C fixed ---
max_iters = [1,2,3,4,5,10,20,100]
C_fixed = 1
train_acc_iter = []
val_acc_iter = []
for max_iter in max_iters:
    model = LogisticRegression(max_iter=max_iter, penalty='l2', solver='liblinear', C=C_fixed)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    train_acc_iter.append(accuracy_score(y_train, y_pred_train))
    val_acc_iter.append(accuracy_score(y_val, y_pred_val))
    print(f"[max_iter sweep] max_iter: {max_iter}, Train Acc: {train_acc_iter[-1]:.4f}, Val Acc: {val_acc_iter[-1]:.4f}")

# --- Plotting and saving plots ---
plt.figure(figsize=(6, 5))

plt.plot(C_values, val_acc_C, marker='o', label='Validation')
plt.plot(C_values, train_acc_C, marker='x', linestyle='--', label='Train')
plt.xscale('log')
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Accuracy vs. C (Max Iterations=500)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logreg_accuracy_vs_C.png")  # Save C experiment plot

plt.figure(figsize=(6, 5))
plt.plot(max_iters, val_acc_iter, marker='o', label='Validation')
plt.plot(max_iters, train_acc_iter, marker='x', linestyle='--', label='Train')
plt.xscale('log')

plt.xlabel('Max Iterations')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Accuracy vs. Max Iterations (C=1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logreg_accuracy_vs_maxiter.png")  # Save max_iter experiment plot

plt.close('all')

# --- Evaluate on test set with best hyperparameters ---
# Find best C (highest validation accuracy)
best_C_idx = np.argmax(val_acc_C)
best_C = C_values[best_C_idx]
print(f"Best C: {best_C} (Validation Acc: {val_acc_C[best_C_idx]:.4f})")

# Find best max_iter (highest validation accuracy)
best_iter_idx = np.argmax(val_acc_iter)
best_max_iter = max_iters[best_iter_idx]
print(f"Best max_iter: {best_max_iter} (Validation Acc: {val_acc_iter[best_iter_idx]:.4f})")

# Evaluate both best C and best max_iter on test set
model_best_C = LogisticRegression(max_iter=500, penalty='l2', solver='liblinear', C=best_C)
model_best_C.fit(X_trainval, y_trainval)
test_acc_best_C = accuracy_score(y_test, model_best_C.predict(X_test))
print(f"Test accuracy with best C ({best_C}): {test_acc_best_C:.4f}")

model_best_iter = LogisticRegression(max_iter=best_max_iter, penalty='l2', solver='liblinear', C=1)
model_best_iter.fit(X_trainval, y_trainval)
test_acc_best_iter = accuracy_score(y_test, model_best_iter.predict(X_test))
print(f"Test accuracy with best max_iter ({best_max_iter}): {test_acc_best_iter:.4f}")

# --- Save best test accuracies to CSV ---
with open("logreg_best_test_accuracies.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Experiment", "Best Hyperparameter", "Test Accuracy"])
    writer.writerow(["C sweep", f"C={best_C}", f"{test_acc_best_C:.4f}"])
    writer.writerow(["max_iter sweep", f"max_iter={best_max_iter}", f"{test_acc_best_iter:.4f}"])


