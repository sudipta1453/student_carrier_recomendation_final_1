import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np 

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

# # import streamlit as st
# import pandas as pd
# import joblib

# # Load your model and scaler (ensure these files are available in the same directory or provide the correct path)
# model_filename = 'voting_classifier_model.pkl'
# scaler_filename = 'scaler.pkl'

# # Load the trained model and scaler
# loaded_model = joblib.load(model_filename)
# scaler = joblib.load(scaler_filename)

# # Create Streamlit GUI for user input
# st.title("Student Data Prediction Form")

# # Input fields
# id = st.slider("Student ID", 1, 1000, 1)
# gender = st.selectbox("Gender", ["Male", "Female"])
# part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
# absence_days = st.slider("Absence Days", 0, 30, 5)
# extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
# weekly_self_study_hours = st.slider("Weekly Self-Study Hours", 0, 50, 10)

# # Subject scores
# math_score = st.slider("Math Score", 0, 100, 70)
# history_score = st.slider("History Score", 0, 100, 70)
# physics_score = st.slider("Physics Score", 0, 100, 70)
# chemistry_score = st.slider("Chemistry Score", 0, 100, 70)
# biology_score = st.slider("Biology Score", 0, 100, 70)
# english_score = st.slider("English Score", 0, 100, 70)
# geography_score = st.slider("Geography Score", 0, 100, 70)

# # Career aspiration (assumed as a score for simplicity)
# #career_aspiration = st.slider("Career Aspiration (1-10)", 1, 10, 5)

# # Data submission and prediction
# if st.button("Submit"):
#     # Convert user input into a DataFrame for the model
#     input_data = pd.DataFrame({
#         'gender': [1 if gender == "Male" else 0],
#         'part_time_job': [1 if part_time_job == "Yes" else 0],
#         'absence_days': [absence_days],
#         'extracurricular_activities': [1 if extracurricular_activities == "Yes" else 0],
#         'weekly_self_study_hours': [weekly_self_study_hours],
#         'math_score': [math_score],
#         'history_score': [history_score],
#         'physics_score': [physics_score],
#         'chemistry_score': [chemistry_score],
#         'biology_score': [biology_score],
#         'english_score': [english_score],
#         'geography_score': [geography_score],
#         'career_aspiration': [career_aspiration]
#     })

#     # Preprocess the input data using the same scaler that was used during training
#     scaled_data = scaler.transform(input_data)

#     # Make prediction
#     prediction = loaded_model.predict(scaled_data)

#     # Display prediction result
#     st.write("Prediction Result:", prediction[0])  # Assuming the model predicts a single value
#     st.success("Prediction made successfully!")


import streamlit as st
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

# Career aspiration mapping
career_map = {
    0: 'Accountant',
    1: 'Artist',
    2: 'Banker',
    3: 'Business Owner',
    4: 'Construction Engineer',
    5: 'Doctor',
    6: 'Game Developer',
    7: 'Government Officer',
    8: 'Lawyer',
    9: 'Social Network Studies',
    10: 'Software Engineer',
    11: 'Stock Investor',
    12: 'Teacher',
    13: 'Writer'
}

# Input fields (excluding career_aspiration since it is now the target)
id = st.slider("Student ID", 1, 1000, 1)  # ID slider from 1 to 1000
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

# Data submission and prediction
if st.button("Submit"):
    # Convert user input into a DataFrame for the model
    input_data = pd.DataFrame({
        'id': [id],
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
        'geography_score': [geography_score]
    })


    # Preprocess the input data using the same scaler that was used during training
    scaled_data = scaler.transform(input_data)

    # Make prediction for career aspiration
    career_aspiration_prediction = loaded_model.predict(scaled_data)[0]

    # Map the numeric prediction to the corresponding career name
    career_name = career_map.get(career_aspiration_prediction, "Unknown Career")

    # Display prediction result for career aspiration
    st.write(f"Predicted Career Aspiration: {career_name}")

    st.success("Prediction made successfully!")

 # ----------- Graphical Output 1: Bar chart of feature importance ----------- #
    feature_names = [
        'gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
        'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score', 
        'chemistry_score', 'biology_score', 'english_score', 'geography_score'
    ]
    
    # Example importance values (replace with actual model feature importances if available)
    feature_importances = np.random.rand(len(feature_names))  # Example random importances
    
    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_importances, color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Career Prediction')

    # Display the bar chart
    st.pyplot(fig1) 

    # ----------- Graphical Output 2: Pie chart of career aspiration distribution ----------- #
    career_counts = np.random.randint(1, 20, size=14)  # Example random counts for demo purposes
    fig2, ax2 = plt.subplots()
    ax2.pie(career_counts, labels=career_map.values(), autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

    # Display the pie chart
    st.pyplot(fig2)

    # ----------- Career Growth Visualization ----------- #
    # Simulate career growth over 5 years
    years = np.arange(1, 6)  # 5 years
    growth_factor = 1.05  # Assuming 5% growth per year
    initial_salary = 50000  # Starting salary

    # Simulate career growth based on study hours and grades
    # Here we assume better grades and higher study hours result in better career growth
    predicted_growth = [initial_salary * (growth_factor ** year) for year in years]

    # Create a line chart for predicted career growth
    fig, ax = plt.subplots()
    ax.plot(years, predicted_growth, marker='o', linestyle='-', color='green')
    ax.set_xlabel('Years')
    ax.set_ylabel('Predicted Salary Growth ($)')
    ax.set_title(f'Predicted Career Growth for {career_name}')

    # Display the line chart
    st.pyplot(fig3)

     # ----------- Additional Growth Insights ----------- #
    # Simulate how grades impact career growth in each year
    grade_factor = (math_score + history_score + physics_score) / 300  # Scale between 0 and 1
    adjusted_growth = [initial_salary * (growth_factor + grade_factor) ** year for year in years]

    # Create a second line chart showing growth with grades considered
    fig2, ax2 = plt.subplots()
    ax2.plot(years, adjusted_growth, marker='o', linestyle='-', color='blue')
    ax2.set_xlabel('Years')
    ax2.set_ylabel('Adjusted Salary Growth ($)')
    ax2.set_title(f'Adjusted Career Growth for {career_name} (Based on Grades)')

    # Display the second chart
    st.pyplot(fig4)

    # ----------- Career Growth Visualization ----------- #
    # Simulate career growth over 5 years
    years = np.arange(1, 6)  # 5 years
    growth_factor = 1.05  # Assuming 5% growth per year
    initial_salary = 50000  # Starting salary

    # Simulate career growth based on study hours and grades
    # Here we assume better grades and higher study hours result in better career growth
    predicted_growth = [initial_salary * (growth_factor ** year) for year in years]

    # Create a line chart for predicted career growth
    fig, ax = plt.subplots()
    ax.plot(years, predicted_growth, marker='o', linestyle='-', color='green')
    ax.set_xlabel('Years')
    ax.set_ylabel('Predicted Salary Growth ($)')
    ax.set_title(f'Predicted Career Growth for {career_name}')

    # Display the line chart
    st.pyplot(fig5)

    # ----------- Additional Growth Insights ----------- #
    # Simulate how grades impact career growth in each year
    grade_factor = (math_score + history_score + physics_score) / 300  # Scale between 0 and 1
    adjusted_growth = [initial_salary * (growth_factor + grade_factor) ** year for year in years]

    # Create a second line chart showing growth with grades considered
    fig2, ax2 = plt.subplots()
    ax2.plot(years, adjusted_growth, marker='o', linestyle='-', color='blue')
    ax2.set_xlabel('Years')
    ax2.set_ylabel('Adjusted Salary Growth ($)')
    ax2.set_title(f'Adjusted Career Growth for {career_name} (Based on Grades)')

    # Display the second chart
    st.pyplot(fig6)

