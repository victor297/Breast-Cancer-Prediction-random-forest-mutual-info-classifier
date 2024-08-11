import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("dataset.csv")

# Remove the extra column 'Unnamed: 32'
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Check for and handle missing values
print("Missing values before handling:")
print(data.isnull().sum())

# Remove rows with missing values
data = data.dropna()

# Check for and handle infinity values
print("Infinity values before handling:")
print((data == float('inf')).sum())

# Replace infinity values with NaN and then drop rows with NaN values
data.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
data.dropna(inplace=True)

# Verify data after cleaning
print("Data after cleaning:")
print(data.info())

# Preprocessing
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Convert diagnosis to binary
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Check the shapes of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Feature selection using Mutual Information
mi = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Selecting top features
top_features = mi_df['Feature'].head(10).tolist()
X_top = X[top_features]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.3, random_state=42)

# Train Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'random_forest_model.pkl')
joblib.dump(top_features, 'features.pkl')  # Save feature names

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print("Classification Report:")
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

#


# Plot Classification Report
plt.figure(figsize=(10, 7))
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T
sns.heatmap(report_df, annot=True, cmap='Blues')
plt.title('Classification Report')
plt.savefig('classification_report.png')
