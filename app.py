import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Saved Models
# -----------------------------
# NOTE: Replace path with your actual model paths
import joblib

clf_model = joblib.load("best_classifier_model.pkl")   # Good Investment Model
reg_model = joblib.load("best_regression_model.pkl")   # Future Price Model
label_encoders = joblib.load("label_encoders.pkl")     # Encoders


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

st.title("🏠 Real Estate Investment Advisor")
st.write("Predict **Good Investment** and **Future Price after 5 Years**")

st.markdown("---")

# -----------------------------
# User Input Form
# -----------------------------
st.subheader("📌 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    State = st.selectbox("State", label_encoders['State'].classes_)
    City = st.selectbox("City", label_encoders['City'].classes_)
    Locality = st.selectbox("Locality", label_encoders['Locality'].classes_)
    Property_Type = st.selectbox("Property Type", label_encoders['Property_Type'].classes_)
    Furnished_Status = st.selectbox("Furnished Status", label_encoders['Furnished_Status'].classes_)

with col2:
    BHK = st.number_input("BHK", min_value=1, max_value=10, value=2)
    Size_in_SqFt = st.number_input("Size (SqFt)", min_value=200, value=1000)
    Price_in_Lakhs = st.number_input("Current Price (Lakhs)", min_value=1, value=50)
    Year_Built = st.number_input("Year Built", min_value=1970, max_value=2025, value=2015)
    Floor_No = st.number_input("Floor No", min_value=0, max_value=50, value=1)

with col3:
    Total_Floors = st.number_input("Total Floors", min_value=1, max_value=50, value=5)
    Nearby_Schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=2)
    Nearby_Hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=1)
    Public_Transport = st.number_input("Public Transport Accessibility (0–10)", min_value=0, max_value=10, value=5)
    Parking_Space = st.number_input("Parking Space", min_value=0, max_value=5, value=1)

Security = st.selectbox("Security", label_encoders['Security'].classes_)
Amenities = st.selectbox("Amenities", label_encoders['Amenities'].classes_)
Facing = st.selectbox("Facing", label_encoders['Facing'].classes_)
Owner_Type = st.selectbox("Owner Type", label_encoders['Owner_Type'].classes_)
Availability_Status = st.selectbox("Availability Status", label_encoders['Availability_Status'].classes_)

# -----------------------------
# Preprocess input
# -----------------------------
if st.button("Predict"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'State': [State],
        'City': [City],
        'Locality': [Locality],
        'Property_Type': [Property_Type],
        'BHK': [BHK],
        'Size_in_SqFt': [Size_in_SqFt],
        'Price_in_Lakhs': [Price_in_Lakhs],
        'Year_Built': [Year_Built],
        'Furnished_Status': [Furnished_Status],
        'Floor_No': [Floor_No],
        'Total_Floors': [Total_Floors],
        'Nearby_Schools': [Nearby_Schools],
        'Nearby_Hospitals': [Nearby_Hospitals],
        'Public_Transport_Accessibility': [Public_Transport],
        'Parking_Space': [Parking_Space],
        'Security': [Security],
        'Amenities': [Amenities],
        'Facing': [Facing],
        'Owner_Type': [Owner_Type],
        'Availability_Status': [Availability_Status]
    })

    # Derived Features
    input_data['Price_per_SqFt'] = Price_in_Lakhs * 100000 / Size_in_SqFt
    input_data['Age_of_Property'] = 2025 - Year_Built

    # Encode Categorical Features
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # -----------------------------
    # Predictions
    # -----------------------------
    good_investment = clf_model.predict(input_data)[0]
    future_price = reg_model.predict(input_data)[0]

    # -----------------------------
    # Show results
    # -----------------------------
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    if good_investment == 1:
        st.success("💰 **This is a GOOD investment property!**")
    else:
        st.error("⚠️ **This is NOT a good investment property.**")

    st.info(f"📈 **Estimated Price After 5 Years:** ₹ {future_price:.2f} Lakhs")


st.markdown("---")
st.caption("Developed with ❤️ using Machine Learning + Streamlit")
