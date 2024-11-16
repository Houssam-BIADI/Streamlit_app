import streamlit as st
import pandas as pd
import pickle
import os
from typing import Dict, Any
import logging

st.write(
    """
# Predicting Heart Disease probality with machine learning using only behavior informations
## This tool can be used to decide if you should see a doctor and do further analyses to detect heart diseases.
### by : Houssam BIADI
"""
)
st.sidebar.subheader("choose your input parameters")


def convert_radio_to_binary(value: str, yes_value: str = "Yes") -> int:
    """Convert radio button response to binary value."""
    return 1 if value == yes_value else 0


def get_age_category(age: int) -> int:
    """Convert age to category value."""
    if age < 18:
        return 0
    elif age <= 24:
        return 1
    elif age >= 80:
        return 13
    else:
        return 3 + (age - 30) // 5


def user_input_features():
    """Collect and process user input features."""
    # Numerical features
    data = {
        "BMI": st.sidebar.slider("Body mass index (kg/m2)", 10, 100, 44),
        "PhysicalHealth": st.sidebar.slider(
            "Days physically ill (past 30 days)", 0, 30, 15
        ),
        "MentalHealth": st.sidebar.slider(
            "Days mentally ill (past 30 days)", 0, 30, 15
        ),
        "SleepTime": st.sidebar.slider(
            "Average sleep hours (24-hour period)", 0, 24, 7
        ),
    }

    # Binary features
    binary_questions = {
        "Smoking": "Have you smoked at least 100 cigarettes in your entire life ?",
        "AlcoholDrinking": "Are you a heavy drinker ( more than 14 drinks per week) ?",
        "Stroke": "Have you ever had a stroke ?",
        "DiffWalking": "Do you have difficulty walking or climbing stairs ?",
        "PhysicalActivity": "Do you do any physical activities at least once a week ?",
        "KidneyDisease": "do you have a Kidney Disease ?",
        "SkinCancer": "Do you have Skin Cancer ?",
        "Asthma": "Do you have Asthma ?",
    }

    for key, question in binary_questions.items():
        data[key] = convert_radio_to_binary(st.sidebar.radio(question, ["Yes", "No"]))

    # Special categories
    data["Sex"] = convert_radio_to_binary(
        st.sidebar.radio("Are you Male or Female ?", ["Male", "Female"]), "Female"
    )

    health_map = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    data["GenHealth"] = health_map[
        st.sidebar.radio(
            "What can you say about your general health ?", list(health_map.keys())
        )
    ]

    data["AgeCategory"] = get_age_category(
        st.sidebar.slider("how old are you ?", 0, 100, 33)
    )

    # Race and Diabetic features (one-hot encoding)
    race = st.sidebar.radio(
        "choose your race ",
        [
            "American Indian/Alaskan Native",
            "Asian",
            "Black",
            "Hispanic",
            "Other",
            "White",
        ],
    )
    diabetic = st.sidebar.radio(
        "Are you Diabetic ", ["No", "Yes", "Only during pregnancy"]
    )

    # Initialize one-hot encoded columns
    for r in ["Asian", "Black", "Hispanic", "Other", "White"]:
        data[f"Race_{r}"] = 1 if race == r else 0

    data |= {
        "Diabetic_No, borderline diabetes": 1 if diabetic == "No" else 0,
        "Diabetic_Yes": 1 if diabetic == "Yes" else 0,
        "Diabetic_Yes (during pregnancy)": (
            1 if diabetic == "Only during pregnancy" else 0
        ),
    }

    return pd.DataFrame(data, index=[0])


# Create features table
df = user_input_features()
st.subheader(
    "We are going to estimate the probability of having a heart desease using the features you entered"
)
st.write(df)
st.header("Predictions")
# Add error handling for model loading
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error(
        "Model file not found. Please ensure 'model.pkl' exists in the application directory."
    )
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()
# Calculate probability of having a heart desease
prediction_proba = model.predict_proba(df)
# show predictions
st.write(
    f"your probabilty of having a heart desease is  **{str(round(prediction_proba[0][1]*100,2))}** %"
)
