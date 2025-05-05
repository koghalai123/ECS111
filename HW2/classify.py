import pandas as pd
import os
import sys 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def save_unique_values_to_csv(df, output_file='unique_values.csv'):
    unique_dict = {}
    
    for column in df.columns:
        unique_values = df[column].unique().tolist()
        unique_dict[column] = unique_values
    
    # Convert to DataFrame (transpose for better CSV structure)
    unique_df = pd.DataFrame.from_dict(unique_dict, orient='index').transpose()
    
    # Save to CSV
    unique_df.to_csv(output_file, index=False)
    print(f"Unique values saved to {output_file}")
    return unique_df


test = pd.read_csv('HW2/adult.test')
test.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv('HW2/adult.data')
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

combined = pd.concat([df,test])
print(combined.head())
unique_df = save_unique_values_to_csv(combined, 'unique_values.csv')

income_class = combined['income']
results_combined = income_class == " >50K"
results_train = results_combined[0:df.shape[0]]
results_test = results_combined[df.shape[0]:combined.shape[0]]
encoded_df = pd.get_dummies(combined.drop(columns=['income'])
)
encoded_train = encoded_df[0:df.shape[0]]
encoded_test = encoded_df[df.shape[0]:encoded_df.shape[0]]

model = LogisticRegression(max_iter=5000, multi_class='auto',)
model.fit(encoded_train, results_train)

# Step 5: Predict on new data
y_pred_train = model.predict(encoded_train)
predictions_train = y_pred_train == results_train
prediction_accuracy_train = np.sum(predictions_train)/predictions_train.shape[0]
print("Prediction Accuracy on Training Data:", prediction_accuracy_train)




y_test = model.predict(encoded_test)
test_predictions = y_test == results_test
test_prediction_accuracy = np.sum(test_predictions)/test_predictions.shape[0]
print("Prediction Accuracy on Test Data:", test_prediction_accuracy)



print(1)








