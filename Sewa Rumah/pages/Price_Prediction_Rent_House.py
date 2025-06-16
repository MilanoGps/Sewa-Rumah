import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🔮 Price Prediction Rent House")

model = joblib.load("model/model_prediksi_rent.pkl")

st.markdown("Isi form di bawah ini untuk memprediksi harga sewa berdasarkan fitur properti:")

with st.form("form_prediksi"):
    city = st.selectbox("Kota", ['Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Delhi', 'Kolkata'])
    furnishing = st.selectbox("Furnishing", ['Unfurnished', 'Semi-Furnished', 'Furnished'])
    area = st.selectbox("Area Type", ['Super Area', 'Carpet Area', 'Built Area'])
    tenant = st.selectbox("Tenant Preferred", ['Bachelors', 'Family', 'Bachelors/Family'])
    contact = st.selectbox("Point of Contact", ['Contact Owner', 'Contact Agent', 'Contact Builder'])
    size = st.number_input("Ukuran (sqft)", 100, 5000, 1000)
    bhk = st.slider("Jumlah BHK", 1, 10, 2)
    bath = st.slider("Jumlah Kamar Mandi", 1, 10, 2)

    submit = st.form_submit_button("Prediksi")

    if submit:
        input_data = pd.DataFrame([{
            'BHK': bhk,
            'Size': size,
            'Bathroom': bath,
            'City': city,
            'Furnishing Status': furnishing,
            'Area Type': area,
            'Tenant Preferred': tenant,
            'Point of Contact': contact
        }])
        pred = model.predict(input_data)
        st.success(f"💰 Prediksi Harga Sewa: ₹ {np.expm1(pred[0]):,.0f}")
