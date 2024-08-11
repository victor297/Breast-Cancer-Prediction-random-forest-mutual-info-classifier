import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and top features
model = joblib.load('random_forest_model.pkl')
top_features = joblib.load('features.pkl')

# Streamlit app
st.title("Breast Cancer Prediction")
st.write("By Falebita Temidayo Janet  20/47cs/01160 and Ibrahim Moshood 20/47cs/01237")

# Option to upload CSV or input manually
st.write("**Choose an input method:**")
input_method = st.radio("Select input method", ('Upload CSV File', 'Manual Input'))

if input_method == 'Upload CSV File':
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Preprocess the data as done earlier
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Convert diagnosis to binary
        X = data[top_features]  # Use only the top features
        y = data['diagnosis']

        # Predict using the loaded model
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y, y_pred)

        # Determine the unique classes in the predictions
        unique_classes = sorted(set(y_pred))

        # Generate the classification report dynamically based on the unique classes
        target_names = ['Benign' if cls == 0 else 'Malignant' for cls in unique_classes]
        report = classification_report(y, y_pred, target_names=target_names, zero_division=0)

        # Display the metrics
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"F1 Score: {f1}")
        st.write("Classification Report:")
        st.text(report)

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to get predictions.")

elif input_method == 'Manual Input':
    # Display input fields for each feature
    inputs = {}
    for feature in top_features:
        inputs[feature] = st.number_input(f"Input {feature}", value=0.0)

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([inputs])

    # Predict using the loaded model
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Display prediction
    if prediction == 0:
        st.write("Prediction: **Benign**")
    else:
        st.write("Prediction: **Malignant**")
    
    st.write(f"Probability of being Benign: {prediction_proba[0]:.2f}")
    st.write(f"Probability of being Malignant: {prediction_proba[1]:.2f}")
