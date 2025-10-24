import streamlit as st
import pickle
import pandas as pd

# Load model and label encoder
model = pickle.load(open("visa_model.pkl", "rb"))
le_ftp = pickle.load(open("le_ftp.pkl", "rb"))

st.title("H1B Visa Approval Prediction")

# --- User Inputs ---
job_titles = ["Software Engineer", "Data Scientist", "Business Analyst", "Project Manager"]
job_title = st.selectbox("Select Job Title", job_titles)

full_time_input = st.selectbox("Full Time or Part Time?", ["Full Time", "Part Time"])
full_time_numeric = 1 if full_time_input == "Full Time" else 0

soc_options = ["15-1256", "15-1244", "11-3021", "13-2011"]
soc_n = st.selectbox("Select SOC_N Code", soc_options)
soc_n_numeric = int(soc_n.replace("-", ""))

# Prevailing Wage (default 100000)
prevailing_wage = st.number_input(
    "Enter Prevailing Wage", 
    min_value=0, 
    value=100000,  # default for full-time
    step=1000
)

year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2025, step=1)

# --- Predict button ---
if st.button("🔍 Predict"):
    input_df = pd.DataFrame({
        "FULL_TIME_POSITION": [full_time_numeric],
        "SOC_N": [soc_n_numeric],
        "PREVAILING_WAGE": [prevailing_wage],
        "YEAR": [year]
    })

    try:
        input_df = input_df[model.feature_names_in_]

        # Prediction
        prediction_numeric = model.predict(input_df)[0]
        label_map = {0: "Denied", 1: "Certified"}
        prediction_label = label_map.get(prediction_numeric, "Unknown")

        # Probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            certified_prob = proba[1] * 100
            denied_prob = proba[0] * 100

            # Color-coded display
            if prediction_label == "Certified":
                st.success(f"Prediction: {prediction_label}")
                st.info(f"Probability - Certified: {certified_prob:.2f}%, Denied: {denied_prob:.2f}%")
            else:
                st.error(f"Prediction: {prediction_label}")
                st.warning(f"Probability - Certified: {certified_prob:.2f}%, Denied: {denied_prob:.2f}%")

        else:
            st.success(f"Prediction: {prediction_label}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
