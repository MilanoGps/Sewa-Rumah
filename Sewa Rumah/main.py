import streamlit as st

st.set_page_config(page_title="House Rent Analysis", layout="wide")

st.title("🏠 House Rent Data Analysis App")
st.markdown("""
Selamat datang di aplikasi interaktif untuk analisis data sewa rumah.  
Gunakan sidebar di kiri untuk memilih analisis:
- **Unsupervised Learning** untuk segmentasi properti berdasarkan fitur numerik.
- **Supervised Learning** untuk prediksi harga sewa menggunakan machine learning.
""")
