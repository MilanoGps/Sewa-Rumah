import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("📊 Supervised Learning - Prediksi Harga Sewa Rumah")

# ===============================
# 1. Visualisasi Sebelum dan Sesudah Pembersihan
# ===============================
st.subheader("1. Visualisasi Sebelum dan Sesudah Pembersihan")
st.markdown("""
Langkah awal dalam supervised learning adalah memahami dan membersihkan data. Di bawah ini ditampilkan:

- **Boxplot sebelum pembersihan**: Menunjukkan sebaran nilai dan outlier.
- **Boxplot setelah pembersihan**: Setelah menghapus duplikat dan outlier.
- **Matriks korelasi**: Untuk melihat hubungan antar fitur numerik.
- **Scatterplot Size vs Rent**: Visualisasi hubungan antara ukuran rumah dan harga sewa.
""")

st.image("assets/boxplot_before_cleaning.png", caption="Boxplot Sebelum Pembersihan")
st.image("assets/boxplot_after_cleaning.png", caption="Boxplot Setelah Pembersihan")
st.image("assets/correlation_matrix_after.png", caption="Matriks Korelasi Setelah Pembersihan")
st.image("assets/scatterplot_size_rent.png", caption="Scatterplot Size vs Rent")

# ===============================
# 2. Korelasi Fitur terhadap Rent
# ===============================
st.subheader("2. Korelasi Fitur terhadap Rent")
st.markdown("""
Menampilkan **persentase korelasi fitur numerik terhadap target** (`Rent`). Korelasi ini membantu kita memahami seberapa besar pengaruh fitur tertentu terhadap harga sewa.
""")
st.image("assets/correlation_percent_to_rent.png", caption="Korelasi Fitur Numerik terhadap Rent")

# ===============================
# 3. Distribusi Fitur Numerik dan Kategorikal
# ===============================
st.subheader("3. Distribusi Fitur Numerik dan Kategorikal")
st.markdown("""
Distribusi fitur digunakan untuk melihat pola sebaran data, deteksi skewness, dan identifikasi nilai dominan. Terdiri dari:

- **Fitur numerik**: `Rent`, `Size`, `BHK`, `Bathroom`.
- **Fitur kategorikal**: `City`, `Furnishing Status`, `Area Type`, `Tenant Preferred`, `Point of Contact`.
""")

# Distribusi fitur numerik
for col in ["rent", "size", "bhk", "bathroom"]:
    st.image(f"assets/dist_{col}.png", caption=f"Distribusi {col.title()}")

# Distribusi fitur kategorikal
for cat in ['city', 'furnishing_status', 'area_type', 'tenant_preferred', 'point_of_contact']:
    st.image(f"assets/barplot_{cat}.png", caption=f"Distribusi {cat.replace('_', ' ').title()}")

# ===============================
# 4. Evaluasi Model
# ===============================
st.subheader("4. Evaluasi Model")
st.markdown("""
Model **Random Forest Regressor** dilatih menggunakan fitur yang telah dibersihkan dan diproses (scaling dan one-hot encoding). Evaluasi dilakukan menggunakan metrik:

- **RMSE, R², MAPE**
- **Visualisasi hasil prediksi vs aktual**
- **Feature importance**: Menunjukkan fitur mana yang paling berkontribusi dalam prediksi.
""")

st.image("assets/barplot_actual_vs_predicted.png", caption="Perbandingan Prediksi vs Aktual (20 Sampel Pertama)")
st.image("assets/feature_importance_supervised.png", caption="Feature Importance dari Random Forest")

# ===============================
# 5. Coba Prediksi Harga Sewa
# ===============================
st.subheader("5. Coba Prediksi Harga Sewa")
st.markdown("""
Silakan isi form berikut untuk mendapatkan **prediksi harga sewa rumah** berdasarkan input properti seperti lokasi, ukuran, furnishing, dan preferensi penyewa. Model akan memberikan estimasi harga sewa bulanan dalam Rupee (₹).
""")

# Load model
model = joblib.load("model/model_prediksi_rent.pkl")

# Form input
with st.form("Prediksi Sewa"):
    city = st.selectbox("Kota", ['Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Delhi', 'Kolkata'])
    furnishing = st.selectbox("Furnishing", ['Unfurnished', 'Semi-Furnished', 'Furnished'])
    area = st.selectbox("Area Type", ['Super Area', 'Carpet Area', 'Built Area'])
    tenant = st.selectbox("Tenant Preferred", ['Bachelors', 'Family', 'Bachelors/Family'])
    contact = st.selectbox("Point of Contact", ['Contact Owner', 'Contact Agent', 'Contact Builder'])
    size = st.number_input("Ukuran (sqft)", 100, 5000, 1000)
    bhk = st.slider("Jumlah BHK", 1, 10, 2)
    bath = st.slider("Jumlah Kamar Mandi", 1, 10, 2)
    
    submitted = st.form_submit_button("Prediksi")
    
    if submitted:
        input_df = pd.DataFrame([{
            'BHK': bhk,
            'Size': size,
            'Bathroom': bath,
            'City': city,
            'Furnishing Status': furnishing,
            'Area Type': area,
            'Tenant Preferred': tenant,
            'Point of Contact': contact
        }])
        pred = model.predict(input_df)
        st.success(f"💰 Prediksi Harga Sewa: ₹ {np.expm1(pred[0]):,.0f}")
