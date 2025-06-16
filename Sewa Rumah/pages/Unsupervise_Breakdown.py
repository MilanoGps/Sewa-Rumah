import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🏘️ Unsupervised Learning - Clustering Rumah Sewa")

# ===============================
# 1. Visualisasi Pembersihan Data
# ===============================
st.subheader("1. Visualisasi Pembersihan Data")
st.markdown("""
Langkah awal dilakukan **eksplorasi dan pembersihan data**:

- **Boxplot Sebelum & Setelah Pembersihan**: Untuk melihat distribusi dan outlier pada fitur numerik (`Rent`, `Size`, `BHK`, `Bathroom`).
- **Matriks Korelasi**: Untuk memahami hubungan antar fitur numerik sebelum dan sesudah pembersihan.
""")
st.image("assets/uns_boxplot_before_cleaning.png", caption="Boxplot Sebelum Pembersihan")
st.image("assets/uns_boxplot_after_cleaning.png", caption="Boxplot Setelah Pembersihan")
st.image("assets/uns_correlation_matrix_before.png", caption="Matriks Korelasi Sebelum")
st.image("assets/uns_correlation_matrix_after.png", caption="Matriks Korelasi Setelah")

# ===============================
# 2. Distribusi Fitur
# ===============================
st.subheader("2. Distribusi Fitur")
st.markdown("""
Distribusi dari fitur numerik (`Rent`, `Size`, `BHK`, dan `Bathroom`) ditampilkan untuk memahami:

- Pola umum data,
- Adanya skewness (kemiringan data),
- Kesesuaian untuk scaling dan clustering.
""")
for col in ["rent", "size", "bhk", "bathroom"]:
    st.image(f"assets/uns_dist_{col}.png", caption=f"Distribusi {col.title()}")

# ===============================
# 3. Penentuan Jumlah Cluster
# ===============================
st.subheader("3. Penentuan Jumlah Cluster")
st.markdown("""
Menggunakan **Metode Elbow** untuk menentukan jumlah cluster optimal (`k`). Grafik menunjukkan nilai **SSE (Sum of Squared Errors)** terhadap jumlah cluster. Titik “tekukan” atau elbow adalah indikasi jumlah cluster yang paling representatif.
""")
st.image("assets/uns_elbow_method.png", caption="Elbow Method")

# ===============================
# 4. Hasil Clustering
# ===============================
st.subheader("4. Hasil Clustering")
st.markdown("""
Setelah menentukan `k=4`, dilakukan:

- **Clustering dengan K-Means**,
- **Visualisasi hasil clustering menggunakan PCA** (mengurangi dimensi ke 2D),
- **Distribusi harga sewa (`Rent`) per cluster**,
- **Jumlah properti pada tiap cluster**.

Ini membantu dalam memahami segmentasi properti berdasarkan fitur-fitur numerik.
""")
st.image("assets/uns_pca_clusters.png", caption="Visualisasi PCA Clustering")
st.image("assets/uns_rent_per_cluster.png", caption="Distribusi Rent per Cluster")
st.image("assets/uns_count_per_cluster.png", caption="Jumlah Properti per Cluster")

# ===============================
# 5. Ringkasan Cluster
# ===============================
st.markdown("---")
st.subheader("5. Ringkasan Cluster")
st.markdown("""
Ringkasan statistik rata-rata (`mean`) dari fitur `Rent`, `Size`, `BHK`, dan `Bathroom` per cluster. Dapat digunakan untuk mengidentifikasi **karakteristik tiap segmen** properti berdasarkan hasil clustering.
""")

st.success("Cluster Summary:")
try:
    cluster_df = pd.read_csv("model/cluster_summary.csv")
    st.dataframe(cluster_df)
except FileNotFoundError:
    st.write("❗ File `cluster_summary.csv` tidak ditemukan. Pastikan file sudah diekspor ke folder `model/`.")
