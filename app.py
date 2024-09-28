import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model (ensure the model is in the same directory or provide the correct path)
model_filename = 'voting_classifier_model.pkl'
loaded_model = joblib.load(model_filename)

# Load the scaler and label encoder (assuming these were saved during training)
scaler = joblib.load('scaler.pkl')  # Ensure this is the same scaler used during training
label_encoder = joblib.load('label_encoder.pkl')  # Ensure this is the same label encoder used during training

# Streamlit app title
st.title("Career Recommendation Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file into a dataframe
    new_data_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data_df.head())  # Display first few rows of the uploaded data

    # Preprocess the new data (apply the same scaling used during training)
    new_data_scaled = scaler.transform(new_data_df)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_data_scaled)

    # Convert predictions back to the original labels using the label encoder
    predictions_labels = label_encoder.inverse_transform(predictions)

    # Display predictions
    st.write("Predictions for the uploaded data:")
    st.write(predictions_labels)

    # Create a DataFrame for the predictions and original data
    result_df = new_data_df.copy()
    result_df['Predicted Career'] = predictions_labels

    # Display the result dataframe
    st.write("Result Data with Predictions:")
    st.write(result_df)

    # Download the results as a CSV
    csv = result_df.to_csv(index=False)
    st.download_button(label="Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')

else:
    st.write("Please upload a CSV file to continue.")
