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

# """ # Upload CSV file
# uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# if uploaded_file is not None:
#     # Read the uploaded file into a dataframe
#     new_data_df = pd.read_csv(uploaded_file)
#     st.write("Uploaded Data:")
#     st.write(new_data_df.head())  # Display first few rows of the uploaded data

#     # Preprocess the new data (apply the same scaling used during training)
#     new_data_scaled = scaler.transform(new_data_df)

#     # Make predictions using the loaded model
#     predictions = loaded_model.predict(new_data_scaled)

#     # Convert predictions back to the original labels using the label encoder
#     predictions_labels = label_encoder.inverse_transform(predictions)

#     # Display predictions
#     st.write("Predictions for the uploaded data:")
#     st.write(predictions_labels)

#     # Create a DataFrame for the predictions and original data
#     result_df = new_data_df.copy()
#     result_df['Predicted Career'] = predictions_labels

#     # Display the result dataframe
#     st.write("Result Data with Predictions:")
#     st.write(result_df)

#     # Download the results as a CSV
#     csv = result_df.to_csv(index=False)
#     st.download_button(label="Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')

# else:
#     st.write("Please upload a CSV file to continue.") """

# import streamlit as st
import pandas as pd
import joblib

# Load your model and scaler (ensure these files are available in the same directory or provide the correct path)
model_filename = 'voting_classifier_model.pkl'
scaler_filename = 'scaler.pkl'

# Load the trained model and scaler
loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Create Streamlit GUI for user input
st.title("Student Data Prediction Form")

# Input fields
id = st.slider("Student ID", 1, 1000, 1)
gender = st.selectbox("Gender", ["Male", "Female"])
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
absence_days = st.slider("Absence Days", 0, 30, 5)
extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
weekly_self_study_hours = st.slider("Weekly Self-Study Hours", 0, 50, 10)

# Subject scores
math_score = st.slider("Math Score", 0, 100, 70)
history_score = st.slider("History Score", 0, 100, 70)
physics_score = st.slider("Physics Score", 0, 100, 70)
chemistry_score = st.slider("Chemistry Score", 0, 100, 70)
biology_score = st.slider("Biology Score", 0, 100, 70)
english_score = st.slider("English Score", 0, 100, 70)
geography_score = st.slider("Geography Score", 0, 100, 70)

# Career aspiration (assumed as a score for simplicity)
career_aspiration = st.slider("Career Aspiration (1-10)", 1, 10, 5)

# Data submission and prediction
if st.button("Submit"):
    # Convert user input into a DataFrame for the model
    input_data = pd.DataFrame({
        'gender': [1 if gender == "Male" else 0],
        'part_time_job': [1 if part_time_job == "Yes" else 0],
        'absence_days': [absence_days],
        'extracurricular_activities': [1 if extracurricular_activities == "Yes" else 0],
        'weekly_self_study_hours': [weekly_self_study_hours],
        'math_score': [math_score],
        'history_score': [history_score],
        'physics_score': [physics_score],
        'chemistry_score': [chemistry_score],
        'biology_score': [biology_score],
        'english_score': [english_score],
        'geography_score': [geography_score],
        'career_aspiration': [career_aspiration]
    })

    # Preprocess the input data using the same scaler that was used during training
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(scaled_data)

    # Display prediction result
    st.write("Prediction Result:", prediction[0])  # Assuming the model predicts a single value
    st.success("Prediction made successfully!")
