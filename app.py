import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Flight Booking Predictor",
    page_icon="✈️",
    layout="centered"
)

# ---------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------

model = joblib.load("model.pkl")

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("✈️ Customer Booking Prediction System")

st.markdown("""
This application predicts whether a customer is likely to complete a flight booking using Machine Learning.
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("About Project")

st.sidebar.info("""
Machine Learning model developed using Random Forest Classifier.

The model predicts customer booking completion behavior based on booking-related features.
""")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("Enter Customer Details")

purchase_lead = st.number_input(
    "Purchase Lead (Days before flight booking)",
    min_value=0,
    max_value=500,
    value=30
)

length_of_stay = st.number_input(
    "Length of Stay (Days)",
    min_value=1,
    max_value=60,
    value=5
)

flight_hour = st.slider(
    "Flight Hour",
    0,
    23,
    12
)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Booking"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'purchase_lead': [purchase_lead],
        'length_of_stay': [length_of_stay],
        'flight_hour': [flight_hour]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1]

    # ---------------------------------------------------
    # OUTPUT SECTION
    # ---------------------------------------------------

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Customer is likely to complete booking")
    else:
        st.error("❌ Customer is unlikely to complete booking")

    # Confidence score
    st.write(f"Prediction Confidence: {probability:.2%}")

    # Progress bar
    st.progress(int(probability * 100))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption("Built with Streamlit, Scikit-Learn, and Python")