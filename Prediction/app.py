
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
with open('Decision_Tree.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit App
def main():
    st.title("COPD Prediction Dashboard")
     
    # User input
    st.sidebar.header("User Input")
     
    age = st.sidebar.slider("Age", 30, 80, 50)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    bmi = st.sidebar.slider("BMI", 10, 40, 25)
    smoking_status = st.sidebar.selectbox('Smoking Status', ["Current", "Former", "Never"])
    biomass_fuel_exposure = st.sidebar.selectbox("Biomass Fuel Exposure", ["Yes", "No"])
    occupational_exposure = st.sidebar.selectbox("Occupational Exposure", ["Yes", "No"])
    family_history = st.sidebar.selectbox("Family History of COPD", ["Yes", "No"])
    air_pollution_level = st.sidebar.slider("Air Pollution Level", 0, 300, 50)
    respiratory_infections = st.sidebar.selectbox("Respiratory Infection in Childhood", ["Yes", "No"])
    pollution_risk_score = st.sidebar.slider("Pollution Risk Score", 0, 100, 50)
    smoking_pollution_interaction = st.sidebar.slider("Smoking Pollution Interaction", 0, 10, 5)
    # Location = st.sidebar.selectbox("Location",['kathmandu','Pokhara','Biratnagar','Lalitpur','Birgunj','Chitwan',"Hetauda","Dharan","Butwal",])
    # Process the input data
    input_data = {
        "Age": [age],
        "Biomass_Fuel_Exposure": [biomass_fuel_exposure],
        "Occupational_Exposure": [occupational_exposure],
        "Family_History_COPD": [family_history],
        "BMI": [bmi],
        "Air_Pollution_Level": [air_pollution_level],
        "Respiratory_Infections_Childhood": [respiratory_infections],
        "Pollution_Risk_Score": [pollution_risk_score],
        "Smoking_Status_encoded": [smoking_status],
        "Gender_": [gender],
        "Smoking_Pollution_interaction": [smoking_pollution_interaction]
    }
    
    # Convert data into a DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Encoding the Data
    input_df['Gender_'] = input_df['Gender_'].map({"Male": 1, 'Female': 0})
    input_df['Smoking_Status_encoded'] = input_df['Smoking_Status_encoded'].map({"Current": 1, 'Former': 0.5, "Never": 0})
    input_df['Biomass_Fuel_Exposure'] = input_df['Biomass_Fuel_Exposure'].map({"Yes": 1, 'No': 0})
    input_df['Occupational_Exposure'] = input_df['Occupational_Exposure'].map({"Yes": 1, 'No': 0})
    input_df['Family_History_COPD'] = input_df['Family_History_COPD'].map({"Yes": 1, 'No': 0})
    input_df['Respiratory_Infections_Childhood'] = input_df['Respiratory_Infections_Childhood'].map({"Yes": 1, 'No': 0}),

    # location_dummies = pd.get_dummies(input_df["Location"],prefix='Location')
    # input_df = pd.concat([input_df, location_dummies], axis = 1)
    
    # input_df.drop('Location',axis=1, inplace=True)

    # Ensure the DataFrame has all required columns
    required_columns = [
        "Age", "Biomass_Fuel_Exposure", "Occupational_Exposure", "Family_History_COPD",
        "BMI", "Air_Pollution_Level", "Respiratory_Infections_Childhood", "Pollution_Risk_Score",
        "Smoking_Status_encoded", "Gender_", "Smoking_Pollution_interaction"
    ]
    
    # Add missing columns if any (set them to default values like 0)
    for column in required_columns:
        if column not in input_df.columns:
            input_df[column] = 0
    
    # Ensure the columns are in the correct order
    input_df = input_df[required_columns]

    # Prediction
    predict = model.predict(input_df)
    if predict[0] == 1:
        st.write("Prediction: COPD Detected")
    else:
        st.write("Prediction: No COPD detected")

if __name__ == "__main__":
    main()
