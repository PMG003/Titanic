# titanic_app.py

import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Encoders ---
@st.cache(allow_output_mutation=True)
def load_model_and_encoders():
    model = joblib.load('random_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')  # assuming you saved these during training
    return model, label_encoders

model, label_encoders = load_model_and_encoders()

# --- App Layout ---
st.title("🎟️ Titanic Passenger Survival Prediction")
st.markdown("Fill the passenger details to predict survival chances!")

# --- Input Form ---
with st.form("passenger_form"):
    passenger_class = st.selectbox("Passenger Class", [1, 2, 3])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age (Years)", 0, 100, 30)
    fare = st.slider("Fare (USD)", 0.0, 600.0, 50.0)
    siblings = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
    parents = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, step=1)
    embarked = st.selectbox("Embarkation Port", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
    
    submit = st.form_submit_button("Predict Survival 🎯")

# --- Prediction Logic ---
if submit:
    try:
        # Preprocess input
        embarked_mapping = {
            "Cherbourg (C)": "C",
            "Queenstown (Q)": "Q",
            "Southampton (S)": "S"
        }
        
        input_data = pd.DataFrame({
            'Pclass': [passenger_class],
            'Sex': [gender],
            'Age': [age],
            'Fare': [fare],
            'SibSp': [siblings],
            'Parch': [parents],
            'Embarked': [embarked_mapping[embarked]]
        })

        # Label Encode necessary columns
        for column in ['Sex', 'Embarked']:
            if column in label_encoders:
                input_data[column] = label_encoders[column].transform(input_data[column].astype(str))

        # Predict
        prediction = model.predict(input_data)[0]

        # Output
        if prediction == 1:
            st.success("🎉 Congratulations! The passenger would have survived.")
        else:
            st.error("💀 Sorry, the passenger would not have survived.")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
