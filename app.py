import streamlit as st
import pandas as pd
import pickle

st.write(
    """
# Predicting Heart Disease probality with machine learning using only behavior informations
## This tool can be used to decide if you should see a doctor and do further analyses to detect heart diseases.
### by : Houssam BIADI
"""
)
st.sidebar.subheader("choose your input parameters")


def user_input_features():
    """This function stores the features that the user is gonna write

    Returns:
        Pandas Dataframe: a pandas dataframe that contains 24 features
    """
    # select numerical features
    Bmi = st.sidebar.slider("Body mass index (kg/m2)", 10, 100, 44)
    PhysicalHealth = st.sidebar.slider(
        "how many days during the past 30 days, you felt physically ill ?", 0, 30, 15
    )
    MentalHealth = st.sidebar.slider(
        "how many days during the past 30 days, you felt mentally ill ?", 0, 30, 15
    )
    SleepTime = st.sidebar.slider(
        "On average, how many hours of sleep do you get in a 24-hour period?", 0, 24, 7
    )
    # select categorical features
    yes_no = ["Yes", "No"]
    Smoking = st.sidebar.radio(
        "Have you smoked at least 100 cigarettes in your entire life ?", yes_no
    )
    Smoking = 1 if Smoking == "Yes" else 0
    AlcoholDrinking = st.sidebar.radio(
        "Are you a heavy drinker ( more than 14 drinks per week) ?", yes_no
    )
    AlcoholDrinking = 1 if AlcoholDrinking == "Yes" else 0
    Stroke = st.sidebar.radio("Have you ever had a stroke ?", yes_no)
    Stroke = 1 if Stroke == "Yes" else 0
    DiffWalking = st.sidebar.radio(
        "Do you have difficulty walking or climbing stairs ?", yes_no
    )
    DiffWalking = 1 if DiffWalking == "Yes" else 0
    PhysicalActivity = st.sidebar.radio(
        "Do you do any physical activities at least once a week ?", yes_no
    )
    PhysicalActivity = 1 if PhysicalActivity == "Yes" else 0
    KidneyDisease = st.sidebar.radio("do you have a Kidney Disease ?", yes_no)
    KidneyDisease = 1 if KidneyDisease == "Yes" else 0
    SkinCancer = st.sidebar.radio("Do you have Skin Cancer ?", yes_no)
    SkinCancer = 1 if SkinCancer == "Yes" else 0
    Asthma = st.sidebar.radio("Do you have Asthma ?", yes_no)
    Asthma = 1 if Asthma == "Yes" else 0
    sex = ["Male", "Female"]
    Sex = st.sidebar.radio("Are you Male or Female ?", sex)
    Sex = 1 if Sex == "Female" else 0
    health = ["Poor", "Fair", "Good", "Very good", "Excellent"]
    GenHealth = st.sidebar.radio("What can you say about your general health ?", health)
    if GenHealth == "Poor":
        GenHealth = 0
    elif GenHealth == "Fair":
        GenHealth = 1
    elif GenHealth == "Good":
        GenHealth = 2
    elif GenHealth == "Very good":
        GenHealth = 3
    else:
        GenHealth = 4

    AgeCategory = st.sidebar.slider("how old are you ?", 0, 100, 33)
    if AgeCategory < 18:
        AgeCategory = 0

    elif 18 <= AgeCategory <= 24:
        AgeCategory = 1

    elif AgeCategory >= 80:
        AgeCategory = 13

    else:
        AgeCategory = 3 + (AgeCategory - 30) // 5
    race_choices = [
        "American Indian/Alaskan Native",
        "Asian",
        "Black",
        "Hispanic",
        "Other",
        "White",
    ]
    Race = st.sidebar.radio("choose your race ", race_choices)
    Diab = ["No", "Yes", "Only during pregnancy"]
    Diabetic = st.sidebar.radio("Are you Diabetic ", Diab)
    data = {
        "BMI": Bmi,
        "Smoking": Smoking,
        "AlcoholDrinking": AlcoholDrinking,
        "Stroke": Stroke,
        "PhysicalHealth": PhysicalHealth,
        "MentalHealth": MentalHealth,
        "DiffWalking": DiffWalking,
        "Sex": Sex,
        "AgeCategory": AgeCategory,
        "PhysicalActivity": PhysicalActivity,
        "GenHealth": GenHealth,
        "SleepTime": SleepTime,
        "Asthma": Asthma,
        "KidneyDisease": KidneyDisease,
        "SkinCancer": SkinCancer,
        "Race_Asian": 0,
        "Race_Black": 0,
        "Race_Hispanic": 0,
        "Race_Other": 0,
        "Race_White": 0,
        "Diabetic_No, borderline diabetes": 0,
        "Diabetic_Yes": 0,
        "Diabetic_Yes (during pregnancy)": 0,
    }
    if Race == "Asian":
        data["Race_Asian"] = 1
    elif Race == "Black":
        data["Race_Black"] = 1
    elif Race == "Hispanic":
        data["Race_Hispanic"] = 1
    elif Race == "Other":
        data["Race_Other"] = 1
    else:
        data["Race_White"] = 1
    if Diabetic == "No":
        data["Diabetic_No, borderline diabetes"] = 1
    elif Diabetic == "Yes":
        data["Diabetic_Yes"] = 1
    else:
        data["Diabetic_Yes (during pregnancy)"] = 1
    features = pd.DataFrame(data, index=[0])
    return features


# Create features table
df = user_input_features()
st.subheader(
    "We are going to estimate the probability of having a heart desease using the features you entered"
)
st.write(df)
st.header("Predictions")
# Load pretrained model
model = pickle.load(open("model.pkl", "rb"))
# Calculate probability of having a heart desease
prediction_proba = model.predict_proba(df)
# show predictions
st.write(
    f"your probabilty of having a heart desease is  **{str(round(prediction_proba[0][1]*100,2))}** %"
)
