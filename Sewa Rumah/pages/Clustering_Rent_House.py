import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Clustering Rent House", layout="wide")
st.title("🏘️ Unsupervised Learning - Clustering Rent House")


# === 1. Rangkuman Statistik per Cluster ===
st.markdown("## 📊 Rangkuman Statistik per Cluster")
if os.path.exists("model/cluster_summary.csv"):
    cluster_df = pd.read_csv("model/cluster_summary.csv", index_col=0)
    st.dataframe(cluster_df)
else:
    st.warning("❌ File 'cluster_summary.csv' tidak ditemukan.")

# === 2. Lihat Data Berdasarkan Cluster ===
st.markdown("## 🔍 Data Detail per Cluster")
if os.path.exists("model/clustered_data.csv"):
    full_df = pd.read_csv("model/clustered_data.csv")
    selected_cluster = st.selectbox("Pilih Cluster:", sorted(full_df["Cluster"].unique()))
    filtered_df = full_df[full_df["Cluster"] == selected_cluster]
    st.write(f"Menampilkan data untuk Cluster {selected_cluster} (total {len(filtered_df)} baris):")
    st.dataframe(filtered_df[['Rent', 'Size', 'BHK', 'Bathroom']].reset_index(drop=True))
else:
    st.warning("❌ File 'clustered_data.csv' tidak ditemukan.")


