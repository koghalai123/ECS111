import pandas as pd
import os
import sys 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



dataL2_libfgs = pd.read_csv('HW2/HW2LogisticRegResultsL2_libfgs.csv')
dataL2_libfgs.columns = ["max_iter","train_acc","test_acc"]

dataL1_liblinear = pd.read_csv('HW2/HW2LogisticRegResultsL1_liblinear.csv')
dataL1_liblinear.columns = ["max_iter","train_acc","test_acc"]

dataL2_liblinear = pd.read_csv('HW2/HW2LogisticRegResultsL2_liblinear.csv')
dataL2_liblinear.columns = ["max_iter","train_acc","test_acc"]

fig = plt.figure()
axes = plt.axes()
axes.plot(dataL2_libfgs['max_iter'], dataL2_libfgs['train_acc'], label='Train Accuracy, Loss: L2, Solver: lbfgs')
axes.plot(dataL1_liblinear['max_iter'], dataL1_liblinear['train_acc'], label='Train Accuracy, Loss: L1, Solver: liblinear ')
axes.plot(dataL2_liblinear['max_iter'], dataL2_liblinear['train_acc'], label='Train Accuracy, Loss: L2, Solver: liblinear ')

axes.set_title("Training Accuracy vs Max Iterations")
axes.set_xlabel("Max Iterations")
axes.set_ylabel("Accuracy")
axes.legend()
axes.set_xscale('log')
plt.savefig('train_accuracy.png')
#axes.set_aspect('equal')
plt.show()

fig = plt.figure()
axes = plt.axes()
axes.plot(dataL2_libfgs['max_iter'], dataL2_libfgs['test_acc'], label='Test Accuracy, Loss: L2, Solver: lbfgs')
axes.plot(dataL1_liblinear['max_iter'], dataL1_liblinear['test_acc'], label='Test Accuracy, Loss: L1, Solver: liblinear ')
axes.plot(dataL2_liblinear['max_iter'], dataL2_liblinear['test_acc'], label='Test Accuracy, Loss: L2, Solver: liblinear ')

axes.set_title("Test Accuracy vs Max Iterations")
axes.set_xlabel("Max Iterations")
axes.set_ylabel("Accuracy")
axes.legend()
axes.set_xscale('log')
plt.savefig('test_accuracy.png')
#axes.set_aspect('equal')
plt.show()



print('done')












