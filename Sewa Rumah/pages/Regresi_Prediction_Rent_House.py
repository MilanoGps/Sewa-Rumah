import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.title("📊 Supervised Learning - Prediksi Harga Sewa Properti")

# === 1. Evaluasi Model ===
st.markdown("---")

# Load hasil prediksi
if os.path.exists("model/prediction_results.csv"):
    pred_df = pd.read_csv("model/prediction_results.csv")  # 'Actual' dan 'Predicted'

    # Hitung metrik evaluasi
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true = pred_df["Actual"]
    y_pred = pred_df["Predicted"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape

    # Penjelasan metrik evaluasi
    st.markdown("### 🧠 Penjelasan Metrik Evaluasi")
    st.markdown("""
Untuk mengukur performa model prediksi harga sewa, digunakan beberapa metrik evaluasi:

- **RMSE (Root Mean Squared Error)**: Menunjukkan rata-rata selisih antara nilai prediksi dan nilai aktual dalam satuan aslinya (harga sewa). Semakin kecil nilai RMSE, semakin baik.
- **R² Score (Koefisien Determinasi)**: Menggambarkan seberapa baik model menjelaskan variasi dalam data. Nilai 1 berarti model sempurna, sedangkan 0 berarti model tidak menjelaskan sama sekali.
- **MAPE (Mean Absolute Percentage Error)**: Rata-rata error dalam bentuk persentase. Semakin kecil MAPE, semakin akurat prediksinya.
- **Akurasi (%)**: Mengukur seberapa dekat prediksi dengan nilai aktual secara rata-rata, dihitung dari `100 - MAPE`.
    """)

    # Tampilkan tabel metrik
    st.markdown("### 📋 Tabel Evaluasi Model")
    eval_df = pd.DataFrame({
        "Metric": ["RMSE", "R² Score", "MAPE", "Akurasi (%)"],
        "Value": [round(rmse, 2), round(r2, 4), round(mape, 2), round(accuracy, 2)]
    })
    st.dataframe(eval_df, use_container_width=True)

    # Scatterplot interaktif
    st.markdown("### 📈 Scatterplot Nilai Aktual vs Prediksi")
    scatter_fig = px.scatter(pred_df, x="Actual", y="Predicted",
                             title="Scatterplot: Aktual vs Prediksi",
                             labels={"Actual": "Harga Sewa Aktual", "Predicted": "Harga Sewa Prediksi"},
                             color_discrete_sequence=["teal"])
    scatter_fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
    scatter_fig.add_shape(
        type='line',
        x0=y_true.min(), y0=y_true.min(),
        x1=y_true.max(), y1=y_true.max(),
        line=dict(color='red', dash='dash'),
    )
    st.plotly_chart(scatter_fig)

else:
    st.warning("❌ File 'prediction_results.csv' tidak ditemukan di folder model/. Harap simpan hasil prediksi dan aktual ke dalam CSV.")
