import streamlit as st
import pandas as pd

st.set_page_config(page_title="House Rent Analysis", layout="wide")

st.title("🏠 House Rent Data Analysis App")
st.markdown("""
Selamat datang di aplikasi interaktif untuk analisis data sewa rumah.  
Gunakan sidebar di kiri untuk memilih analisis:
- **Unsupervised Learning** untuk segmentasi properti berdasarkan KMeans.
- **Supervised Learning** untuk prediksi harga sewa menggunakan random forest regresi.
""")

# === Tampilkan Data dari CSV ===
st.subheader("📄 Preview Dataset")

try:
    df = pd.read_csv("model/House_Rent_Dataset.csv")
    st.dataframe(df.head(100))  # tampilkan 100 baris pertama agar ringan
except FileNotFoundError:
    st.warning("❌ File 'House_Rent_Dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sesuai.")
