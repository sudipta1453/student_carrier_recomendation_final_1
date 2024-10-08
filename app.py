# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import numpy as np 

# # Load the trained model (ensure the model is in the same directory or provide the correct path)
# model_filename = 'voting_classifier_model.pkl'
# loaded_model = joblib.load(model_filename)

# # Load the scaler and label encoder (assuming these were saved during training)
# scaler = joblib.load('scaler.pkl')  # Ensure this is the same scaler used during training
# label_encoder = joblib.load('label_encoder.pkl')  # Ensure this is the same label encoder used during training

# # Streamlit app title
# st.title("Career Recommendation Prediction")

# # """ # Upload CSV file
# # uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# # if uploaded_file is not None:
# #     # Read the uploaded file into a dataframe
# #     new_data_df = pd.read_csv(uploaded_file)
# #     st.write("Uploaded Data:")
# #     st.write(new_data_df.head())  # Display first few rows of the uploaded data

# #     # Preprocess the new data (apply the same scaling used during training)
# #     new_data_scaled = scaler.transform(new_data_df)

# #     # Make predictions using the loaded model
# #     predictions = loaded_model.predict(new_data_scaled)

# #     # Convert predictions back to the original labels using the label encoder
# #     predictions_labels = label_encoder.inverse_transform(predictions)

# #     # Display predictions
# #     st.write("Predictions for the uploaded data:")
# #     st.write(predictions_labels)

# #     # Create a DataFrame for the predictions and original data
# #     result_df = new_data_df.copy()
# #     result_df['Predicted Career'] = predictions_labels

# #     # Display the result dataframe
# #     st.write("Result Data with Predictions:")
# #     st.write(result_df)

# #     # Download the results as a CSV
# #     csv = result_df.to_csv(index=False)
# #     st.download_button(label="Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')

# # else:
# #     st.write("Please upload a CSV file to continue.") """

# # # import streamlit as st
# # import pandas as pd
# # import joblib

# # # Load your model and scaler (ensure these files are available in the same directory or provide the correct path)
# # model_filename = 'voting_classifier_model.pkl'
# # scaler_filename = 'scaler.pkl'

# # # Load the trained model and scaler
# # loaded_model = joblib.load(model_filename)
# # scaler = joblib.load(scaler_filename)

# # # Create Streamlit GUI for user input
# # st.title("Student Data Prediction Form")

# # # Input fields
# # id = st.slider("Student ID", 1, 1000, 1)
# # gender = st.selectbox("Gender", ["Male", "Female"])
# # part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
# # absence_days = st.slider("Absence Days", 0, 30, 5)
# # extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
# # weekly_self_study_hours = st.slider("Weekly Self-Study Hours", 0, 50, 10)

# # # Subject scores
# # math_score = st.slider("Math Score", 0, 100, 70)
# # history_score = st.slider("History Score", 0, 100, 70)
# # physics_score = st.slider("Physics Score", 0, 100, 70)
# # chemistry_score = st.slider("Chemistry Score", 0, 100, 70)
# # biology_score = st.slider("Biology Score", 0, 100, 70)
# # english_score = st.slider("English Score", 0, 100, 70)
# # geography_score = st.slider("Geography Score", 0, 100, 70)

# # # Career aspiration (assumed as a score for simplicity)
# # #career_aspiration = st.slider("Career Aspiration (1-10)", 1, 10, 5)

# # # Data submission and prediction
# # if st.button("Submit"):
# #     # Convert user input into a DataFrame for the model
# #     input_data = pd.DataFrame({
# #         'gender': [1 if gender == "Male" else 0],
# #         'part_time_job': [1 if part_time_job == "Yes" else 0],
# #         'absence_days': [absence_days],
# #         'extracurricular_activities': [1 if extracurricular_activities == "Yes" else 0],
# #         'weekly_self_study_hours': [weekly_self_study_hours],
# #         'math_score': [math_score],
# #         'history_score': [history_score],
# #         'physics_score': [physics_score],
# #         'chemistry_score': [chemistry_score],
# #         'biology_score': [biology_score],
# #         'english_score': [english_score],
# #         'geography_score': [geography_score],
# #         'career_aspiration': [career_aspiration]
# #     })

# #     # Preprocess the input data using the same scaler that was used during training
# #     scaled_data = scaler.transform(input_data)

# #     # Make prediction
# #     prediction = loaded_model.predict(scaled_data)

# #     # Display prediction result
# #     st.write("Prediction Result:", prediction[0])  # Assuming the model predicts a single value
# #     st.success("Prediction made successfully!")


# import streamlit as st
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

# # Career aspiration mapping
# career_map = {
#     0: 'Accountant',
#     1: 'Artist',
#     2: 'Banker',
#     3: 'Business Owner',
#     4: 'Construction Engineer',
#     5: 'Doctor',
#     6: 'Game Developer',
#     7: 'Government Officer',
#     8: 'Lawyer',
#     9: 'Social Network Studies',
#     10: 'Software Engineer',
#     11: 'Stock Investor',
#     12: 'Teacher',
#     13: 'Writer'
# }

# # Input fields (excluding career_aspiration since it is now the target)
# id = st.slider("Student ID", 1, 1000, 1)  # ID slider from 1 to 1000
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

# # Data submission and prediction
# if st.button("Submit"):
#     # Convert user input into a DataFrame for the model
#     input_data = pd.DataFrame({
#         'id': [id],
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
#         'geography_score': [geography_score]
#     })


#     # Preprocess the input data using the same scaler that was used during training
#     scaled_data = scaler.transform(input_data)

#     # Make prediction for career aspiration
#     career_aspiration_prediction = loaded_model.predict(scaled_data)[0]

#     # Map the numeric prediction to the corresponding career name
#     predicted_career_name = career_map.get(career_aspiration_prediction, "Unknown Career")

#     # Display prediction result for career aspiration
#     st.write(f"Predicted Career Aspiration: {predicted_career_name}")

#     st.success("Prediction made successfully!")

# #  # ----------- Graphical Output 1: Bar chart of feature importance ----------- #
# #     feature_names = [
# #         'gender', 'part_time_job', 'absence_days', 'extracurricular_activities', 
# #         'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score', 
# #         'chemistry_score', 'biology_score', 'english_score', 'geography_score'
# #     ]
    
# #     # Example importance values (replace with actual model feature importances if available)
# #     feature_importances = np.random.rand(len(feature_names))  # Example random importances
    
# #     fig, ax = plt.subplots()
# #     ax.barh(feature_names, feature_importances, color='skyblue')
# #     ax.set_xlabel('Importance')
# #     ax.set_title('Feature Importance in Career Prediction')

# #     # Display the bar chart
# #     st.pyplot(fig) 

# #     # ----------- Graphical Output 2: Pie chart of career aspiration distribution ----------- #
# #     career_counts = np.random.randint(1, 20, size=14)  # Example random counts for demo purposes
# #     fig2, ax2 = plt.subplots()
# #     ax2.pie(career_counts, labels=career_map.values(), autopct='%1.1f%%', startangle=90)
# #     ax2.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

# #     # Display the pie chart
# #     st.pyplot(fig1)

# #     # ----------- Career Growth Visualization ----------- #
# #     # Simulate career growth over 5 years
# #     years = np.arange(1, 6)  # 5 years
# #     growth_factor = 1.05  # Assuming 5% growth per year
# #     initial_salary = 50000  # Starting salary

# #     # Simulate career growth based on study hours and grades
# #     # Here we assume better grades and higher study hours result in better career growth
# #     predicted_growth = [initial_salary * (growth_factor ** year) for year in years]

# #     # Create a line chart for predicted career growth
# #     fig, ax = plt.subplots()
# #     ax.plot(years, predicted_growth, marker='o', linestyle='-', color='green')
# #     ax.set_xlabel('Years')
# #     ax.set_ylabel('Predicted Salary Growth ($)')
# #     ax.set_title(f'Predicted Career Growth for {career_name}')

# #     # Display the line chart
# #     st.pyplot(fig)

# #      # ----------- Additional Growth Insights ----------- #
# #     # Simulate how grades impact career growth in each year
# #     grade_factor = (math_score + history_score + physics_score) / 300  # Scale between 0 and 1
# #     adjusted_growth = [initial_salary * (growth_factor + grade_factor) ** year for year in years]

# #     # Create a second line chart showing growth with grades considered
# #     fig2, ax2 = plt.subplots()
# #     ax2.plot(years, adjusted_growth, marker='o', linestyle='-', color='blue')
# #     ax2.set_xlabel('Years')
# #     ax2.set_ylabel('Adjusted Salary Growth ($)')
# #     ax2.set_title(f'Adjusted Career Growth for {career_name} (Based on Grades)')

# #     # Display the second chart
# #     st.pyplot(fig1)

# #     # ----------- Career Growth Visualization ----------- #
# #     # Simulate career growth over 5 years
# #     years = np.arange(1, 6)  # 5 years
# #     growth_factor = 1.05  # Assuming 5% growth per year
# #     initial_salary = 50000  # Starting salary

# #     # Simulate career growth based on study hours and grades
# #     # Here we assume better grades and higher study hours result in better career growth
# #     predicted_growth = [initial_salary * (growth_factor ** year) for year in years]

# #     # Create a line chart for predicted career growth
# #     fig, ax = plt.subplots()
# #     ax.plot(years, predicted_growth, marker='o', linestyle='-', color='green')
# #     ax.set_xlabel('Years')
# #     ax.set_ylabel('Predicted Salary Growth ($)')
# #     ax.set_title(f'Predicted Career Growth for {career_name}')

# #     # Display the line chart
# #     st.pyplot(fig)

# #     # ----------- Additional Growth Insights ----------- #
# #     # Simulate how grades impact career growth in each year
# #     grade_factor = (math_score + history_score + physics_score) / 300  # Scale between 0 and 1
# #     adjusted_growth = [initial_salary * (growth_factor + grade_factor) ** year for year in years]

# #     # Create a second line chart showing growth with grades considered
# #     fig2, ax2 = plt.subplots()
# #     ax2.plot(years, adjusted_growth, marker='o', linestyle='-', color='blue')
# #     ax2.set_xlabel('Years')
# #     ax2.set_ylabel('Adjusted Salary Growth ($)')
# #     ax2.set_title(f'Adjusted Career Growth for {career_name} (Based on Grades)')

# #     # Display the second chart
# #     st.pyplot(fig1)

# # ----------- Career Growth Visualization ----------- #
#     # Simulate career growth over 5 years
#     years = np.arange(1, 6)  # 5 years
#     growth_factor = 1.05  # Assuming 5% growth per year
#     initial_salary = 50000  # Starting salary

#     # Simulate career growth based on study hours and grades
#     # Here we assume better grades and higher study hours result in better career growth
#     predicted_growth = [initial_salary * (growth_factor ** year) for year in years]

#     # Adjusted growth based on grades
#     grade_factor = (math_score + history_score + physics_score) / 300  # Scale between 0 and 1
#     adjusted_growth = [initial_salary * (growth_factor + grade_factor) ** year for year in years]

#     # Create subplots: 2 rows, 3 columns for 6 charts
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

#     # Chart 1: Predicted career growth (line chart)
#     axes[0, 0].plot(years, predicted_growth, marker='o', linestyle='-', color='green')
#     axes[0, 0].set_title(f'Predicted Career Growth for {predicted_career_name}')
#     axes[0, 0].set_xlabel('Years')
#     axes[0, 0].set_ylabel('Predicted Salary Growth ($)')

#     # Chart 2: Adjusted career growth based on grades (line chart)
#     axes[0, 1].plot(years, adjusted_growth, marker='o', linestyle='-', color='blue')
#     axes[0, 1].set_title(f'Adjusted Career Growth (Grades)')
#     axes[0, 1].set_xlabel('Years')
#     axes[0, 1].set_ylabel('Adjusted Salary Growth ($)')

#     # Chart 3: Scatter plot of study hours vs math score
#     axes[0, 2].scatter(input_data['weekly_self_study_hours'], input_data['math_score'], color='red')
#     axes[0, 2].set_title('Study Hours vs Math Score')
#     axes[0, 2].set_xlabel('Weekly Study Hours')
#     axes[0, 2].set_ylabel('Math Score')

#     # Chart 4: Histogram of study hours
#     axes[1, 0].hist(input_data['weekly_self_study_hours'], bins=5, color='purple')
#     axes[1, 0].set_title('Histogram of Study Hours')
#     axes[1, 0].set_xlabel('Weekly Study Hours')
#     axes[1, 0].set_ylabel('Frequency')

#     # Chart 5: Bar chart for feature importances (random example values)
#     feature_importances = np.random.rand(12)  # Example random importances
#     feature_names = ['gender', 'part_time_job', 'absence_days', 'extracurricular_activities',
#                      'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
#                      'chemistry_score', 'biology_score', 'english_score', 'geography_score']
#     axes[1, 1].barh(feature_names, feature_importances, color='orange')
#     axes[1, 1].set_title('Feature Importance')

#     # Chart 6: Pie chart for career aspiration distribution (simulated)
#     career_aspiration_distribution = np.random.randint(1, 20, size=14)  # Random counts
#     axes[1, 2].pie(career_aspiration_distribution, labels=career_map.values(), autopct='%1.1f%%', startangle=90)
#     axes[1, 2].set_title('Career Aspiration Distribution')

#     # Adjust layout to prevent overlap
#     plt.tight_layout()

#     # Display the subplots
#     st.pyplot(fig)

# #     import streamlit as st

# # # Career Map with details on criteria, subjects, and colleges
# # career_details = {
# #     'Accountant': {
# #         'eligibility': "Bachelor's degree in Accounting or related field",
# #         'subjects': ["Accounting", "Finance", "Taxation", "Economics"],
# #         'colleges': ["University of Chicago", "Harvard University", "Stanford University"]
# #     },
# #     'Artist': {
# #         'eligibility': "Bachelor's degree in Fine Arts or relevant experience",
# #         'subjects': ["Visual Arts", "Art History", "Digital Art", "Sculpture"],
# #         'colleges': ["Rhode Island School of Design", "California Institute of the Arts", "School of the Art Institute of Chicago"]
# #     },
# #     'Banker': {
# #         'eligibility': "Bachelor's degree in Finance or Business Administration",
# #         'subjects': ["Finance", "Economics", "Banking", "Business Law"],
# #         'colleges': ["Wharton School of the University of Pennsylvania", "London School of Economics", "MIT Sloan"]
# #     },
# #     'Business Owner': {
# #         'eligibility': "No formal requirements, but MBA is recommended",
# #         'subjects': ["Business Management", "Marketing", "Entrepreneurship", "Finance"],
# #         'colleges': ["Harvard Business School", "Stanford Graduate School of Business", "INSEAD"]
# #     },
# #     'Construction Engineer': {
# #         'eligibility': "Bachelor's degree in Civil Engineering or Construction Management",
# #         'subjects': ["Civil Engineering", "Construction Technology", "Project Management", "Structural Engineering"],
# #         'colleges': ["Massachusetts Institute of Technology (MIT)", "Stanford University", "University of California, Berkeley"]
# #     },
# #     'Doctor': {
# #         'eligibility': "MBBS or equivalent, followed by specialization",
# #         'subjects': ["Biology", "Chemistry", "Physics", "Anatomy", "Medical Science"],
# #         'colleges': ["Harvard Medical School", "Johns Hopkins University", "Mayo Clinic Alix School of Medicine"]
# #     },
# #     'Game Developer': {
# #         'eligibility': "Bachelor's degree in Computer Science or Game Design",
# #         'subjects': ["Programming", "Game Design", "Graphics", "AI for Games"],
# #         'colleges': ["University of Southern California", "Carnegie Mellon University", "DigiPen Institute of Technology"]
# #     },
# #     'Government Officer': {
# #         'eligibility': "Bachelor's degree and pass civil services examination",
# #         'subjects': ["Public Administration", "Political Science", "Law", "Economics"],
# #         'colleges': ["Indian Institutes of Public Administration", "London School of Economics", "Harvard University"]
# #     },
# #     'Lawyer': {
# #         'eligibility': "Bachelor's degree in Law (LLB), followed by bar examination",
# #         'subjects': ["Law", "Constitutional Law", "Criminal Law", "Civil Law"],
# #         'colleges': ["Harvard Law School", "Yale Law School", "Stanford Law School"]
# #     },
# #     'Social Network Studies': {
# #         'eligibility': "Bachelor's degree in Media Studies or Sociology",
# #         'subjects': ["Sociology", "Media Studies", "Communication", "Psychology"],
# #         'colleges': ["University of California, Berkeley", "Stanford University", "Columbia University"]
# #     },
# #     'Software Engineer': {
# #         'eligibility': "Bachelor's degree in Computer Science or Software Engineering",
# #         'subjects': ["Computer Science", "Software Development", "Data Structures", "Algorithms"],
# #         'colleges': ["Massachusetts Institute of Technology (MIT)", "Stanford University", "Carnegie Mellon University"]
# #     },
# #     'Stock Investor': {
# #         'eligibility': "Bachelor's degree in Finance or Economics, certifications like CFA",
# #         'subjects': ["Finance", "Economics", "Stock Markets", "Investment Analysis"],
# #         'colleges': ["Wharton School of the University of Pennsylvania", "London School of Economics", "Harvard University"]
# #     },
# #     'Teacher': {
# #         'eligibility': "Bachelor's degree in Education or specialization in a subject",
# #         'subjects': ["Education", "Pedagogy", "Psychology", "Subject Specialization"],
# #         'colleges': ["University of Oxford", "Harvard Graduate School of Education", "Stanford University"]
# #     },
# #     'Writer': {
# #         'eligibility': "Bachelor's degree in Literature, Creative Writing, or Journalism",
# #         'subjects': ["Creative Writing", "Literature", "Journalism", "Communications"],
# #         'colleges': ["Iowa Writers' Workshop", "Columbia University", "University of Cambridge"]
# #     }
# # }

# # # # Streamlit app
# # # def career_guidance_app():
# # #     st.title("Career Path Guidance")
    
# # #     # Career selection dropdown
# # #     career_choice = st.selectbox("Choose a Profession", list(career_details.keys()))
    
# # #     if career_choice:
# # #         # Display career details
# # #         st.header(f"Path to becoming a {career_choice}")
        
# # #         st.subheader("Eligibility Criteria")
# # #         st.write(career_details[career_choice]['eligibility'])
        
# # #         st.subheader("Recommended Subjects")
# # #         st.write(", ".join(career_details[career_choice]['subjects']))
        
# # #         st.subheader("Recommended Colleges")
# # #         st.write(", ".join(career_details[career_choice]['colleges']))

# # # # Run the Streamlit app
# # # if __name__ == '__main__':
# # #     career_guidance_app()

# # # # Streamlit app
# # # def career_guidance_app():
# # #    # st.set_page_config(page_title="Career Path Guidance", layout="wide")

# # #     # Sidebar for career selection
# # #     st.sidebar.title("Career Guidance")
# # #     st.sidebar.markdown("### Choose a Profession:")
    
# # #     # Career selection dropdown
# # #     career_choice = st.sidebar.selectbox("Select a Career", list(career_details.keys()))
    
# # #     # Title and introduction
# # #     st.title("Career Path Guidance")
# # #     st.markdown("""
# # #     Welcome to the Career Path Guidance app! Here, you can explore different career paths, understand the eligibility criteria, 
# # #     subjects to focus on, and recommended colleges for each profession. Simply select a career from the sidebar to get started.
# # #     """)
    
# # #     # Career information display
# # #     if career_choice:
# # #         # Displaying career details in a structured manner
# # #         st.header(f"Path to Becoming a {career_choice}")
        
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.subheader("Eligibility Criteria")
# # #             st.write(career_details[career_choice]['eligibility'])
        
# # #         with col2:
# # #             st.subheader("Recommended Subjects")
# # #             st.write(", ".join(career_details[career_choice]['subjects']))
        
# # #         st.subheader("Top Recommended Colleges")
# # #         st.write(", ".join(career_details[career_choice]['colleges']))
        
# # #         # Add a footer with additional information
# # #         st.markdown("""
# # #         **Note**: These are just examples of paths, and there are many different routes to success. 
# # #         It is always best to research the specific requirements of your country or preferred institution.
# # #         """)

# # # # Run the Streamlit app
# # # if __name__ == '__main__':
# # #     career_guidance_app()

# # # Streamlit app
# # def career_guidance_app():
# #     #st.set_page_config(page_title="Career Path Guidance", layout="wide")

# #     # Sidebar with search and selection options
# #     st.sidebar.title("Career Guidance")
# #     st.sidebar.markdown("### Choose or Search for a Profession:")

# #     # Search bar in sidebar
# #     search_input = st.sidebar.text_input("Search for a Career", "")
    
# #     # Filter careers by search query
# #     filtered_careers = {k: v for k, v in career_details.items() if search_input.lower() in k.lower()}
    
# #     # If search returns results, show in dropdown; otherwise, use the full list
# #     if filtered_careers:
# #         career_choice = st.sidebar.selectbox("Careers Found", list(filtered_careers.keys()))
# #     else:
# #         career_choice = st.sidebar.selectbox("Careers", list(career_details.keys()))

# #     # Title and introduction
# #     st.title("Career Path Guidance")
# #     st.markdown("""
# #     Welcome to the Career Path Guidance app! Explore various career paths, discover eligibility criteria, 
# #     subjects to focus on, recommended colleges, salary ranges, career growth prospects, and job demand for each profession.
# #     Simply select a career from the sidebar or search for one.
# #     """)

# #     # Career information display
# #     if career_choice:
# #         # Display career details in a structured manner
# #         st.header(f"Path to Becoming a {career_choice}")
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             st.subheader("Eligibility Criteria")
# #             st.write(career_details[career_choice]['eligibility'])
        
# #         with col2:
# #             st.subheader("Recommended Subjects")
# #             st.write(", ".join(career_details[career_choice]['subjects']))
        
# #         st.subheader("Top Recommended Colleges")
# #         colleges = career_details[career_choice]['colleges']
# #         for college, url in colleges.items():
# #             st.markdown(f"- [{college}]({url})")  # External links to college websites
        
# #         # Add salary range
# #         st.subheader("Average Salary Range")
# #         st.write(career_details[career_choice]['salary_range'])

# #         # Add career growth prospects
# #         st.subheader("Career Growth Prospects")
# #         st.write(career_details[career_choice]['career_growth'])
        
# #         # Add job demand
# #         st.subheader("Job Demand")
# #         st.write(career_details[career_choice]['job_demand'])
        
# #         # Add a footer with additional information
# #         st.markdown("""
# #         **Note**: These are just examples of paths, and there are many different routes to success. 
# #         It is always best to research the specific requirements of your country or preferred institution.
# #         """)

# # # Run the Streamlit app
# # if __name__ == '__main__':
# #     career_guidance_app()

# #     import streamlit as st

# # # Updated Career Map with salary ranges, growth, and job demand
# # career_details = {
# #     'Accountant': {
# #         'eligibility': "Bachelor's degree in Accounting or related field",
# #         'subjects': ["Accounting", "Finance", "Taxation", "Economics"],
# #         'colleges': {
# #             "University of Chicago": "https://www.uchicago.edu",
# #             "Harvard University": "https://www.harvard.edu",
# #             "Stanford University": "https://www.stanford.edu"
# #         },
# #         'salary': "$50,000 - $100,000 per year",
# #         'growth': "Stable growth with advancements in financial technology",
# #         'demand': "Moderate demand, especially for certified professionals"
# #     },
# #     'Artist': {
# #         'eligibility': "Bachelor's degree in Fine Arts or relevant experience",
# #         'subjects': ["Visual Arts", "Art History", "Digital Art", "Sculpture"],
# #         'colleges': {
# #             "Rhode Island School of Design": "https://www.risd.edu",
# #             "California Institute of the Arts": "https://www.calarts.edu",
# #             "School of the Art Institute of Chicago": "https://www.saic.edu"
# #         },
# #         'salary': "$30,000 - $70,000 per year",
# #         'growth': "Moderate, growth driven by digital art and online media",
# #         'demand': "Variable demand, high in digital sectors"
# #     },
# #     'Banker': {
# #         'eligibility': "Bachelor's degree in Finance or Business Administration",
# #         'subjects': ["Finance", "Economics", "Banking", "Business Law"],
# #         'colleges': {
# #             "Wharton School of the University of Pennsylvania": "https://www.wharton.upenn.edu",
# #             "London School of Economics": "https://www.lse.ac.uk",
# #             "MIT Sloan": "https://mitsloan.mit.edu"
# #         },
# #         'salary': "$60,000 - $150,000 per year",
# #         'growth': "High growth in investment banking and fintech sectors",
# #         'demand': "High demand for investment and risk management roles"
# #     },
# #     'Doctor': {
# #         'eligibility': "MBBS or equivalent, followed by specialization",
# #         'subjects': ["Biology", "Chemistry", "Physics", "Anatomy", "Medical Science"],
# #         'colleges': {
# #             "Harvard Medical School": "https://meded.hms.harvard.edu",
# #             "Johns Hopkins University": "https://www.hopkinsmedicine.org",
# #             "Mayo Clinic Alix School of Medicine": "https://college.mayo.edu"
# #         },
# #         'salary': "$150,000 - $300,000 per year",
# #         'growth': "Strong growth due to aging population and healthcare demand",
# #         'demand': "Very high demand, especially for specialized fields"
# #     },
# #     'Software Engineer': {
# #         'eligibility': "Bachelor's degree in Computer Science or Software Engineering",
# #         'subjects': ["Computer Science", "Software Development", "Data Structures", "Algorithms"],
# #         'colleges': {
# #             "MIT": "https://www.csail.mit.edu",
# #             "Stanford University": "https://cs.stanford.edu",
# #             "Carnegie Mellon University": "https://www.cs.cmu.edu"
# #         },
# #         'salary': "$80,000 - $180,000 per year",
# #         'growth': "Rapid growth due to increasing demand for software solutions",
# #         'demand': "Extremely high demand, especially in AI, cybersecurity, and cloud computing"
# #     },
# #     # Add more professions similarly...
# # }

# # # Streamlit app
# # def career_guidance_app():
# #     #st.set_page_config(page_title="Career Path Guidance", layout="wide")

# #     # Sidebar with search and selection options
# #     st.sidebar.title("Career Guidance")
# #     st.sidebar.markdown("### Choose or Search for a Profession:")

# #     # Search bar in sidebar
# #     search_input = st.sidebar.text_input("Search for a Career", "")
    
# #     # Filter careers by search query
# #     filtered_careers = {k: v for k, v in career_details.items() if search_input.lower() in k.lower()}
    
# #     # If search returns results, show in dropdown; otherwise, use the full list
# #     if filtered_careers:
# #         career_choice = st.sidebar.selectbox("Careers Found", list(filtered_careers.keys()))
# #     else:
# #         career_choice = st.sidebar.selectbox("Careers", list(career_details.keys()))

# #     # Title and introduction
# #     st.title("Career Path Guidance")
# #     st.markdown("""
# #     Welcome to the Career Path Guidance app! Explore various career paths, discover eligibility criteria, 
# #     subjects to focus on, recommended colleges, and additional metrics like salary ranges, career growth, 
# #     and job demand. Simply select a career from the sidebar or search for one.
# #     """)

# #     # Career information display
# #     if career_choice:
# #         # Display career details in a structured manner
# #         st.header(f"Path to Becoming a {career_choice}")
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             st.subheader("Eligibility Criteria")
# #             st.write(career_details[career_choice]['eligibility'])
        
# #         with col2:
# #             st.subheader("Recommended Subjects")
# #             st.write(", ".join(career_details[career_choice]['subjects']))
        
# #         st.subheader("Top Recommended Colleges")
# #         colleges = career_details[career_choice]['colleges']
# #         for college, url in colleges.items():
# #             st.markdown(f"- [{college}]({url})")  # External links to college websites
        
# #         # Display additional metrics: salary, growth, demand
# #         st.subheader("Career Metrics")
        
# #         st.markdown(f"**Average Salary Range**: {career_details[career_choice]['salary']}")
# #         st.markdown(f"**Career Growth**: {career_details[career_choice]['growth']}")
# #         st.markdown(f"**Job Demand**: {career_details[career_choice]['demand']}")
        
# #         # Add a footer with additional information
# #         st.markdown("""
# #         **Note**: These are just examples of paths, and there are many different routes to success. 
# #         It is always best to research the specific requirements of your country or preferred institution.
# #         """)

# # # Run the Streamlit app
# # if __name__ == '__main__':
# #     career_guidance_app()


# import streamlit as st

# # Set page configuration at the very beginning of the script
# #st.set_page_config(page_title="Career Path Guidance", layout="wide")

# # Updated Career Map with salary ranges, growth, and job demand for all professions
# career_details = {
#     'Accountant': {
#         'eligibility': "Bachelor's degree in Accounting or related field",
#         'subjects': ["Accounting", "Finance", "Taxation", "Economics"],
#         'colleges': {
#             "University of Chicago": "https://www.uchicago.edu",
#             "Harvard University": "https://www.harvard.edu",
#             "Stanford University": "https://www.stanford.edu"
#         },
#         'salary': "$50,000 - $100,000 per year",
#         'growth': "Stable growth with advancements in financial technology",
#         'demand': "Moderate demand, especially for certified professionals"
#     },
#     'Artist': {
#         'eligibility': "Bachelor's degree in Fine Arts or relevant experience",
#         'subjects': ["Visual Arts", "Art History", "Digital Art", "Sculpture"],
#         'colleges': {
#             "Rhode Island School of Design": "https://www.risd.edu",
#             "California Institute of the Arts": "https://www.calarts.edu",
#             "School of the Art Institute of Chicago": "https://www.saic.edu"
#         },
#         'salary': "$30,000 - $70,000 per year",
#         'growth': "Moderate, growth driven by digital art and online media",
#         'demand': "Variable demand, high in digital sectors"
#     },
#     'Banker': {
#         'eligibility': "Bachelor's degree in Finance or Business Administration",
#         'subjects': ["Finance", "Economics", "Banking", "Business Law"],
#         'colleges': {
#             "Wharton School of the University of Pennsylvania": "https://www.wharton.upenn.edu",
#             "London School of Economics": "https://www.lse.ac.uk",
#             "MIT Sloan": "https://mitsloan.mit.edu"
#         },
#         'salary': "$60,000 - $150,000 per year",
#         'growth': "High growth in investment banking and fintech sectors",
#         'demand': "High demand for investment and risk management roles"
#     },
#     'Business Owner': {
#         'eligibility': "No formal requirements, but an MBA is recommended",
#         'subjects': ["Business Management", "Marketing", "Entrepreneurship", "Finance"],
#         'colleges': {
#             "Harvard Business School": "https://www.hbs.edu",
#             "Stanford Graduate School of Business": "https://www.gsb.stanford.edu",
#             "INSEAD": "https://www.insead.edu"
#         },
#         'salary': "Varies widely depending on business success",
#         'growth': "High potential for growth but high risk as well",
#         'demand': "Entrepreneurship is growing in the digital and tech sectors"
#     },
#     'Construction Engineer': {
#         'eligibility': "Bachelor's degree in Civil Engineering or Construction Management",
#         'subjects': ["Civil Engineering", "Construction Technology", "Project Management", "Structural Engineering"],
#         'colleges': {
#             "MIT": "https://www.mit.edu",
#             "Stanford University": "https://www.stanford.edu",
#             "University of California, Berkeley": "https://www.berkeley.edu"
#         },
#         'salary': "$70,000 - $120,000 per year",
#         'growth': "Steady growth as infrastructure projects increase",
#         'demand': "High demand due to urban development and infrastructure projects"
#     },
#     'Doctor': {
#         'eligibility': "MBBS or equivalent, followed by specialization",
#         'subjects': ["Biology", "Chemistry", "Physics", "Anatomy", "Medical Science"],
#         'colleges': {
#             "Harvard Medical School": "https://meded.hms.harvard.edu",
#             "Johns Hopkins University": "https://www.hopkinsmedicine.org",
#             "Mayo Clinic Alix School of Medicine": "https://college.mayo.edu"
#         },
#         'salary': "$150,000 - $300,000 per year",
#         'growth': "Strong growth due to aging population and healthcare demand",
#         'demand': "Very high demand, especially for specialized fields"
#     },
#     'Game Developer': {
#         'eligibility': "Bachelor's degree in Computer Science or Game Design",
#         'subjects': ["Programming", "Game Design", "Graphics", "AI for Games"],
#         'colleges': {
#             "University of Southern California": "https://www.usc.edu",
#             "Carnegie Mellon University": "https://www.cmu.edu",
#             "DigiPen Institute of Technology": "https://www.digipen.edu"
#         },
#         'salary': "$60,000 - $120,000 per year",
#         'growth': "Fast growth in mobile, AR, VR, and console gaming",
#         'demand': "High demand, especially in indie game development and VR/AR"
#     },
#     'Government Officer': {
#         'eligibility': "Bachelor's degree and pass civil services examination",
#         'subjects': ["Public Administration", "Political Science", "Law", "Economics"],
#         'colleges': {
#             "Indian Institute of Public Administration": "https://www.iipa.org.in",
#             "London School of Economics": "https://www.lse.ac.uk",
#             "Harvard University": "https://www.harvard.edu"
#         },
#         'salary': "$50,000 - $120,000 per year (depending on rank and role)",
#         'growth': "Stable, with advancement through ranks",
#         'demand': "High demand, especially for roles in public policy and administration"
#     },
#     'Lawyer': {
#         'eligibility': "Bachelor's degree in Law (LLB), followed by bar examination",
#         'subjects': ["Law", "Constitutional Law", "Criminal Law", "Civil Law"],
#         'colleges': {
#             "Harvard Law School": "https://hls.harvard.edu",
#             "Yale Law School": "https://law.yale.edu",
#             "Stanford Law School": "https://law.stanford.edu"
#         },
#         'salary': "$70,000 - $200,000 per year",
#         'growth': "Moderate growth with specialization in corporate, environmental, or tech law",
#         'demand': "High demand for corporate lawyers, IP lawyers, and environmental lawyers"
#     },
#     'Social Network Studies': {
#         'eligibility': "Bachelor's degree in Media Studies or Sociology",
#         'subjects': ["Sociology", "Media Studies", "Communication", "Psychology"],
#         'colleges': {
#             "University of California, Berkeley": "https://www.berkeley.edu",
#             "Stanford University": "https://www.stanford.edu",
#             "Columbia University": "https://www.columbia.edu"
#         },
#         'salary': "$40,000 - $90,000 per year",
#         'growth': "Growing with the rise of social media and digital communities",
#         'demand': "High demand for roles in social media analysis, online communities, and digital marketing"
#     },
#     'Software Engineer': {
#         'eligibility': "Bachelor's degree in Computer Science or Software Engineering",
#         'subjects': ["Computer Science", "Software Development", "Data Structures", "Algorithms"],
#         'colleges': {
#             "MIT": "https://www.csail.mit.edu",
#             "Stanford University": "https://cs.stanford.edu",
#             "Carnegie Mellon University": "https://www.cs.cmu.edu"
#         },
#         'salary': "$80,000 - $180,000 per year",
#         'growth': "Rapid growth due to increasing demand for software solutions",
#         'demand': "Extremely high demand, especially in AI, cybersecurity, and cloud computing"
#     },
#     'Stock Investor': {
#         'eligibility': "Bachelor's degree in Finance or Economics, certifications like CFA",
#         'subjects': ["Finance", "Economics", "Stock Markets", "Investment Analysis"],
#         'colleges': {
#             "Wharton School of the University of Pennsylvania": "https://www.wharton.upenn.edu",
#             "London School of Economics": "https://www.lse.ac.uk",
#             "Harvard University": "https://www.harvard.edu"
#         },
#         'salary': "Varies widely, high earnings potential based on investments",
#         'growth': "High growth potential but highly volatile",
#         'demand': "Moderate demand, especially for certified financial analysts"
#     },
#     'Teacher': {
#         'eligibility': "Bachelor's degree in Education or specialization in a subject",
#         'subjects': ["Education", "Pedagogy", "Psychology", "Subject Specialization"],
#         'colleges': {
#             "University of Oxford": "https://www.ox.ac.uk",
#             "Harvard Graduate School of Education": "https://www.gse.harvard.edu",
#             "Stanford University": "https://www.stanford.edu"
#         },
#         'salary': "$40,000 - $70,000 per year",
#         'growth': "Stable, with opportunities to specialize or move into administration",
#         'demand': "High demand, especially in STEM subjects and special education"
#     },
#     'Writer': {
#         'eligibility': "Bachelor's degree in Literature, Creative Writing, or Journalism",
#         'subjects': ["Creative Writing", "Literature", "Journalism", "Communications"],
#         'colleges': {
#             "Iowa Writers' Workshop": "https://writersworkshop.uiowa.edu",
#             "Columbia University": "https://www.columbia.edu",
#             "University of Cambridge": "https://www.cam.ac.uk"
#         },
#         'salary': "$30,000 - $100,000 per year",
#         'growth': "Moderate growth with increasing digital content opportunities",
#         'demand': "High demand for content creators, journalists, and technical writers"
#     }
# }

# # Example career prediction logic
# predicted_career = predicted_career_name  # This is an example of a predicted career

# # Streamlit app
# def career_guidance_app():
#     st.title("Career Path Guidance")

#     # Display the predicted career aspiration
#     st.write(f"Predicted Career Aspiration: {predicted_career}")

#     # Get the predicted career's details
#     if predicted_career in career_details:
#         career_info = career_details[predicted_career]
        
#         st.header(f"Path to Becoming a {predicted_career}")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Eligibility Criteria")
#             st.write(career_info['eligibility'])
        
#         with col2:
#             st.subheader("Recommended Subjects")
#             st.write(", ".join(career_info['subjects']))
        
#         st.subheader("Top Recommended Colleges")
#         colleges = career_info['colleges']
#         for college, url in colleges.items():
#             st.markdown(f"- [{college}]({url})")  # External links to college websites
        
#         # Display additional metrics: salary, growth, demand
#         st.subheader("Career Metrics")
        
#         st.markdown(f"**Average Salary Range**: {career_info['salary']}")
#         st.markdown(f"**Career Growth**: {career_info['growth']}")
#         st.markdown(f"**Job Demand**: {career_info['demand']}")
        
#         # Add a footer with additional information
#         st.markdown("""
#         **Note**: These are just examples of paths, and there are many different routes to success. 
#         It is always best to research the specific requirements of your country or preferred institution.
#         """)
#     else:
#         st.write("Career details not available for the predicted career.")



# # Fetch and display career guidance details
#     if predicted_career_name in career_details:
#         career_info = career_details[predicted_career_name]
        
#         st.header(f"Career Path for {predicted_career_name}")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Eligibility Criteria")
#             st.write(career_info['eligibility'])
        
#         with col2:
#             st.subheader("Recommended Subjects")
#             st.write(", ".join(career_info['subjects']))
        
#         st.subheader("Top Recommended Colleges")
#         colleges = career_info['colleges']
#         for college, url in colleges.items():
#             st.markdown(f"- [{college}]({url})")  # External links to college websites
        
#         # Display additional metrics: salary, growth, demand
#         st.subheader("Career Metrics")
        
#         st.markdown(f"**Average Salary Range**: {career_info['salary']}")
#         st.markdown(f"**Career Growth**: {career_info['growth']}")
#         st.markdown(f"**Job Demand**: {career_info['demand']}")
        
#         st.success("Career details fetched successfully!")
#     else:
#         st.error("Career details not available.")

#         # Run the Streamlit app
# if __name__ == '__main__':
#     career_guidance_app()
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler (ensure these files are available in the same directory or provide the correct path)
model_filename = 'voting_classifier_model.pkl'
scaler_filename = 'scaler.pkl'

# Load the trained model and scaler
loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Career aspiration mapping (this maps the predicted number to the career name)
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

# # Career details dictionary
# career_details = {
#     'Accountant': {
#         'eligibility': "Bachelor's degree in Accounting or related field",
#         'subjects': ["Accounting", "Finance", "Taxation", "Economics"],
#         'colleges': {
#             "University of Chicago": "https://www.uchicago.edu",
#             "Harvard University": "https://www.harvard.edu",
#             "Stanford University": "https://www.stanford.edu"
#         },
#         'salary': "$50,000 - $100,000 per year",
#         'growth': "Stable growth with advancements in financial technology",
#         'demand': "Moderate demand, especially for certified professionals"
#     },
#     'Software Engineer': {
#         'eligibility': "Bachelor's degree in Computer Science or Software Engineering",
#         'subjects': ["Computer Science", "Software Development", "Data Structures", "Algorithms"],
#         'colleges': {
#             "MIT": "https://www.csail.mit.edu",
#             "Stanford University": "https://cs.stanford.edu",
#             "Carnegie Mellon University": "https://www.cs.cmu.edu"
#         },
#         'salary': "$80,000 - $180,000 per year",
#         'growth': "Rapid growth due to increasing demand for software solutions",
#         'demand': "Extremely high demand, especially in AI, cybersecurity, and cloud computing"
#     },
#     # Add more career details here...
# }
# Career details dictionary with more career paths
career_details = {
    'Accountant': {
        'eligibility': "Bachelor's degree in Accounting or related field",
        'subjects': ["Accounting", "Finance", "Taxation", "Economics"],
        'colleges': {
            "University of Chicago": "https://www.uchicago.edu",
            "Harvard University": "https://www.harvard.edu",
            "Stanford University": "https://www.stanford.edu"
        },
        'salary': "$50,000 - $100,000 per year",
        'growth': "Stable growth with advancements in financial technology",
        'demand': "Moderate demand, especially for certified professionals"
    },
    'Artist': {
        'eligibility': "Bachelor's degree in Fine Arts or relevant experience",
        'subjects': ["Visual Arts", "Art History", "Digital Art", "Sculpture"],
        'colleges': {
            "Rhode Island School of Design": "https://www.risd.edu",
            "California Institute of the Arts": "https://www.calarts.edu",
            "School of the Art Institute of Chicago": "https://www.saic.edu"
        },
        'salary': "$30,000 - $70,000 per year",
        'growth': "Moderate, growth driven by digital art and online media",
        'demand': "Variable demand, high in digital sectors"
    },
    'Banker': {
        'eligibility': "Bachelor's degree in Finance or Business Administration",
        'subjects': ["Finance", "Economics", "Banking", "Business Law"],
        'colleges': {
            "Wharton School of the University of Pennsylvania": "https://www.wharton.upenn.edu",
            "London School of Economics": "https://www.lse.ac.uk",
            "MIT Sloan": "https://mitsloan.mit.edu"
        },
        'salary': "$60,000 - $150,000 per year",
        'growth': "High growth in investment banking and fintech sectors",
        'demand': "High demand for investment and risk management roles"
    },
    'Business Owner': {
        'eligibility': "No formal requirements, but an MBA is recommended",
        'subjects': ["Business Management", "Marketing", "Entrepreneurship", "Finance"],
        'colleges': {
            "Harvard Business School": "https://www.hbs.edu",
            "Stanford Graduate School of Business": "https://www.gsb.stanford.edu",
            "INSEAD": "https://www.insead.edu"
        },
        'salary': "Varies widely depending on business success",
        'growth': "High potential for growth but high risk as well",
        'demand': "Entrepreneurship is growing in the digital and tech sectors"
    },
    'Construction Engineer': {
        'eligibility': "Bachelor's degree in Civil Engineering or Construction Management",
        'subjects': ["Civil Engineering", "Construction Technology", "Project Management", "Structural Engineering"],
        'colleges': {
            "MIT": "https://www.mit.edu",
            "Stanford University": "https://www.stanford.edu",
            "University of California, Berkeley": "https://www.berkeley.edu"
        },
        'salary': "$70,000 - $120,000 per year",
        'growth': "Steady growth as infrastructure projects increase",
        'demand': "High demand due to urban development and infrastructure projects"
    },
    'Doctor': {
        'eligibility': "MBBS or equivalent, followed by specialization",
        'subjects': ["Biology", "Chemistry", "Physics", "Anatomy", "Medical Science"],
        'colleges': {
            "Harvard Medical School": "https://meded.hms.harvard.edu",
            "Johns Hopkins University": "https://www.hopkinsmedicine.org",
            "Mayo Clinic Alix School of Medicine": "https://college.mayo.edu"
        },
        'salary': "$150,000 - $300,000 per year",
        'growth': "Strong growth due to aging population and healthcare demand",
        'demand': "Very high demand, especially for specialized fields"
    },
    'Game Developer': {
        'eligibility': "Bachelor's degree in Computer Science or Game Design",
        'subjects': ["Programming", "Game Design", "Graphics", "AI for Games"],
        'colleges': {
            "University of Southern California": "https://www.usc.edu",
            "Carnegie Mellon University": "https://www.cmu.edu",
            "DigiPen Institute of Technology": "https://www.digipen.edu"
        },
        'salary': "$60,000 - $120,000 per year",
        'growth': "Fast growth in mobile, AR, VR, and console gaming",
        'demand': "High demand, especially in indie game development and VR/AR"
    },
    'Government Officer': {
        'eligibility': "Bachelor's degree and pass civil services examination",
        'subjects': ["Public Administration", "Political Science", "Law", "Economics"],
        'colleges': {
            "Indian Institute of Public Administration": "https://www.iipa.org.in",
            "London School of Economics": "https://www.lse.ac.uk",
            "Harvard University": "https://www.harvard.edu"
        },
        'salary': "$50,000 - $120,000 per year (depending on rank and role)",
        'growth': "Stable, with advancement through ranks",
        'demand': "High demand, especially for roles in public policy and administration"
    },
    'Lawyer': {
        'eligibility': "Bachelor's degree in Law (LLB), followed by bar examination",
        'subjects': ["Law", "Constitutional Law", "Criminal Law", "Civil Law"],
        'colleges': {
            "Harvard Law School": "https://hls.harvard.edu",
            "Yale Law School": "https://law.yale.edu",
            "Stanford Law School": "https://law.stanford.edu"
        },
        'salary': "$70,000 - $200,000 per year",
        'growth': "Moderate growth with specialization in corporate, environmental, or tech law",
        'demand': "High demand for corporate lawyers, IP lawyers, and environmental lawyers"
    },
    'Social Network Studies': {
        'eligibility': "Bachelor's degree in Media Studies or Sociology",
        'subjects': ["Sociology", "Media Studies", "Communication", "Psychology"],
        'colleges': {
            "University of California, Berkeley": "https://www.berkeley.edu",
            "Stanford University": "https://www.stanford.edu",
            "Columbia University": "https://www.columbia.edu"
        },
        'salary': "$40,000 - $90,000 per year",
        'growth': "Growing with the rise of social media and digital communities",
        'demand': "High demand for roles in social media analysis, online communities, and digital marketing"
    },
    'Software Engineer': {
        'eligibility': "Bachelor's degree in Computer Science or Software Engineering",
        'subjects': ["Computer Science", "Software Development", "Data Structures", "Algorithms"],
        'colleges': {
            "MIT": "https://www.csail.mit.edu",
            "Stanford University": "https://cs.stanford.edu",
            "Carnegie Mellon University": "https://www.cs.cmu.edu"
        },
        'salary': "$80,000 - $180,000 per year",
        'growth': "Rapid growth due to increasing demand for software solutions",
        'demand': "Extremely high demand, especially in AI, cybersecurity, and cloud computing"
    },
    'Stock Investor': {
        'eligibility': "Bachelor's degree in Finance or Economics, certifications like CFA",
        'subjects': ["Finance", "Economics", "Stock Markets", "Investment Analysis"],
        'colleges': {
            "Wharton School of the University of Pennsylvania": "https://www.wharton.upenn.edu",
            "London School of Economics": "https://www.lse.ac.uk",
            "Harvard University": "https://www.harvard.edu"
        },
        'salary': "Varies widely, high earnings potential based on investments",
        'growth': "High growth potential but highly volatile",
        'demand': "Moderate demand, especially for certified financial analysts"
    },
    'Teacher': {
        'eligibility': "Bachelor's degree in Education or specialization in a subject",
        'subjects': ["Education", "Pedagogy", "Psychology", "Subject Specialization"],
        'colleges': {
            "University of Oxford": "https://www.ox.ac.uk",
            "Harvard Graduate School of Education": "https://www.gse.harvard.edu",
            "Stanford University": "https://www.stanford.edu"
        },
        'salary': "$40,000 - $70,000 per year",
        'growth': "Stable, with opportunities to specialize or move into administration",
        'demand': "High demand, especially in STEM subjects and special education"
    },
    'Writer': {
        'eligibility': "Bachelor's degree in Literature, Creative Writing, or Journalism",
        'subjects': ["Creative Writing", "Literature", "Journalism", "Communications"],
        'colleges': {
            "Iowa Writers' Workshop": "https://writersworkshop.uiowa.edu",
            "Columbia University": "https://www.columbia.edu",
            "University of Cambridge": "https://www.cam.ac.uk"
        },
        'salary': "$30,000 - $100,000 per year",
        'growth': "Moderate growth with increasing digital content opportunities",
        'demand': "High demand for content creators, journalists, and technical writers"
    }
}

# Streamlit app for user input
st.title("Student Career Prediction and Guidance")

# Input fields (these will be used to predict the career)
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

    # Make prediction for career aspiration using the model
    predicted_career_code = loaded_model.predict(scaled_data)[0]

    # Map the numeric prediction to the corresponding career name
    predicted_career_name = career_map.get(predicted_career_code, "Unknown Career")

    # Display prediction result for career aspiration
    st.write(f"Predicted Career Aspiration: {predicted_career_name}")

    # Fetch and display career guidance details
    if predicted_career_name in career_details:
        career_info = career_details[predicted_career_name]
        
        st.header(f"Career Path for {predicted_career_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Eligibility Criteria")
            st.write(career_info['eligibility'])
        
        with col2:
            st.subheader("Recommended Subjects")
            st.write(", ".join(career_info['subjects']))
        
        st.subheader("Top Recommended Colleges")
        colleges = career_info['colleges']
        for college, url in colleges.items():
            st.markdown(f"- [{college}]({url})")  # External links to college websites
        
        # Display additional metrics: salary, growth, demand
        st.subheader("Career Metrics")
        
        st.markdown(f"**Average Salary Range**: {career_info['salary']}")
        st.markdown(f"**Career Growth**: {career_info['growth']}")
        st.markdown(f"**Job Demand**: {career_info['demand']}")
        
        st.success("Career details fetched successfully!")
    else:
        st.error("Career details not available.")



