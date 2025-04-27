import streamlit as st
import pandas as pd
import joblib

import os

@st.cache_resource
def load_model():
    if os.path.exists('random_forest_model.pkl'):
        return joblib.load('random_forest_model.pkl')
    else:
        st.error("⚠️ Model file not found! Please upload 'random_forest_model.pkl'.")
        st.stop()

# App Configuration
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

st.markdown("""
Predict whether a passenger survived the Titanic disaster using a trained **Random Forest** model.
Adjust the passenger details below and click **Predict Survival**!
""")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

model = load_model()

# Input Features
st.subheader("🎟️ Passenger Details")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age (Years)", 0.0, 100.0, 30.0)
    fare = st.slider("Fare (USD)", 0.0, 600.0, 32.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    embarked = st.selectbox("Embarkation Port", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])

# Feature Engineering
family_size = sibsp + parch
embarked_code = embarked[0]  # Extracts 'C', 'Q', or 'S'

# Convert categorical features
input_data = {
    'Pclass': pclass,
    'Sex': 1 if sex == "Male" else 0,
    'Age': age,
    'Fare': fare,
    'FamilySize': family_size,
    'Embarked_1': 1 if embarked_code == 'C' else 0,
    'Embarked_2': 1 if embarked_code == 'Q' else 0,
    'Embarked_3': 1 if embarked_code == 'S' else 0,
    'Title_1': 1 if title == "Mr" else 0,
    'Title_2': 1 if title == "Mrs" else 0,
    'Title_3': 1 if title == "Miss" else 0,
    'Title_4': 1 if title == "Master" else 0
}

# Ensure DataFrame columns match model's expectation
expected_columns = [
    'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize',
    'Embarked_1', 'Embarked_2', 'Embarked_3',
    'Title_1', 'Title_2', 'Title_3', 'Title_4'
]
input_df = pd.DataFrame([input_data], columns=expected_columns)

# Prediction
if st.button("🔮 Predict Survival"):
    try:
        prediction = model.predict(input_df)[0]
        survival_status = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
        st.success(f"**Prediction:** {survival_status}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by [Your Name] | Model: `Random Forest Titanic Survival`")
