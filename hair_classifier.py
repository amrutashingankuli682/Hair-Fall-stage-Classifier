import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.datasets import load_files
from IPython.display import display, HTML

# Load the dataset
df = pd.read_csv('hair_loss.csv')

# Display the first 5 rows
print(df.head())

# Separate features and target variable
X = df.drop('hair_fall', axis=1)  
y = df['hair_fall']  

# Convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy()

# Instantiate the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform the training data
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='poly')
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
