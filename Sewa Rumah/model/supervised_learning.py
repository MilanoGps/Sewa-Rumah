# Supervised Learning: Prediksi Harga Sewa Rumah

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 1. Load Dataset ==========
df = pd.read_csv("House_Rent_Dataset.csv")

# ========== 2. Data Understanding ==========
print("==== Informasi Dataset ====")
print(df.info())
print("\n==== Statistik Deskriptif ====")
print(df.describe())
print("\n==== 5 Data Teratas ====")
print(df.head())

# Drop kolom 'Posted On' karena tidak relevan
df.drop(columns=['Posted On'], inplace=True)

# ========== 3. Visualisasi Sebelum Pembersihan ==========
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Rent', 'Size', 'BHK', 'Bathroom']])
plt.title("Boxplot Sebelum Pembersihan")
plt.savefig("boxplot_before_cleaning.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df[['Rent', 'Size', 'BHK', 'Bathroom']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Sebelum Pembersihan")
plt.savefig("correlation_matrix_before.png")
plt.show()

# ========== 4. Pembersihan Data ==========
print("\nJumlah duplikat:", df.duplicated().sum())
df = df.drop_duplicates()

df = df[df['Rent'] < 500000]
df = df[df['Size'] < 5000]

print("\nMissing values:")
print(df.isnull().sum())
if df.isnull().sum().sum() > 0:
    print("\n❗ Ada nilai kosong. Menghapus baris yang mengandung missing values...")
    df = df.dropna()
else:
    print("✅ Tidak ada missing values.")

# ========== 5. Visualisasi Setelah Pembersihan ==========
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Rent', 'Size', 'BHK', 'Bathroom']])
plt.title("Boxplot Setelah Pembersihan")
plt.savefig("boxplot_after_cleaning.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df[['Rent', 'Size', 'BHK', 'Bathroom']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Setelah Pembersihan")
plt.savefig("correlation_matrix_after.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='Size', y='Rent', hue='City')
plt.title("Size vs Rent")
plt.savefig("scatterplot_size_rent.png")
plt.show()

# ========== 5a. Korelasi Numerik terhadap Rent dalam Persentase ==========
corr_matrix = df[['Rent', 'Size', 'BHK', 'Bathroom']].corr()
corr_with_rent = corr_matrix['Rent'].drop('Rent') * 100
corr_with_rent = corr_with_rent.sort_values(ascending=False)

print("\n=== Korelasi Fitur Numerik terhadap Rent (dalam %) ===")
print(corr_with_rent.round(2))

plt.figure(figsize=(8, 5))
sns.barplot(x=corr_with_rent.values, y=corr_with_rent.index, palette='coolwarm')
plt.title("Persentase Korelasi Fitur Numerik terhadap Rent")
plt.xlabel("Korelasi (%)")
plt.ylabel("Fitur")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("correlation_percent_to_rent.png")
plt.show()

# ========== 5b. Visualisasi Distribusi Fitur ==========
# Numerik
numerical_features = ['Rent', 'Size', 'BHK', 'Bathroom']
for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=True, color='skyblue', edgecolor='black', linewidth=0.5)
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"dist_{col.lower()}.png")
    plt.show()

# Kategorikal
categorical_features = ['City', 'Furnishing Status', 'Area Type', 'Tenant Preferred', 'Point of Contact']
for col in categorical_features:
    plt.figure(figsize=(10, 5))
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, palette='Set2')
    plt.title(f'Jumlah per Kategori: {col}')
    plt.xlabel(col)
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"barplot_{col.lower().replace(' ', '_')}.png")
    plt.show()

# ========== 6. Modeling ==========
X = df[['BHK', 'Size', 'Bathroom', 'City', 'Furnishing Status',
        'Area Type', 'Tenant Preferred', 'Point of Contact']]
y = np.log1p(df['Rent'])  # log(1 + Rent)

categorical = ['City', 'Furnishing Status', 'Area Type', 'Tenant Preferred', 'Point of Contact']
numerical = ['BHK', 'Size', 'Bathroom']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# ========== 7. Evaluasi Model ==========

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Pastikan folder 'model' ada
os.makedirs("model", exist_ok=True)

# Buat DataFrame hasil prediksi lengkap
actual_values = np.expm1(y_test).values  # Invers log1p
full_result_df = pd.DataFrame({
    'Actual': actual_values,
    'Predicted': y_pred
})
full_result_df.to_csv("prediction_results.csv", index=False)
print("✅ Hasil prediksi disimpan di 'prediction_results.csv'")

# ===== Barplot 20 Sampel Pertama =====
comparison_df = full_result_df.head(20).reset_index(drop=True)
plt.figure(figsize=(12, 6))
comparison_df.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title("Perbandingan Harga Sewa: Aktual vs Prediksi (20 Sampel)")
plt.xlabel("Index Sampel")
plt.ylabel("Rent")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("barplot_actual_vs_predicted.png")
plt.close()

# ===== Scatterplot Prediksi vs Aktual =====
plt.figure(figsize=(8, 6))
sns.scatterplot(x=actual_values, y=y_pred)
plt.title("Scatterplot: Aktual vs Prediksi Harga Sewa")
plt.xlabel("Harga Aktual")
plt.ylabel("Harga Prediksi")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatterplot_actual_vs_predicted.png")
plt.close()
print("✅ Scatterplot disimpan ke 'scatterplot_actual_vs_predicted.png'")

# ===== Metrik Evaluasi =====
rmse = np.sqrt(mean_squared_error(actual_values, y_pred))
r2 = r2_score(actual_values, y_pred)
mape = np.mean(np.abs((actual_values - y_pred) / actual_values)) * 100
accuracy = 100 - mape

# Tampilkan
print("\n==== Evaluasi Model ====")
print(f"RMSE           : {rmse:.2f}")
print(f"R² Score       : {r2:.4f}")
print(f"MAPE           : {mape:.2f}%")
print(f"Akurasi (%)    : {accuracy:.2f}%")

# Simpan ke CSV
eval_df = pd.DataFrame({
    "Metric": ["RMSE", "R²", "MAPE", "Akurasi (%)"],
    "Value": [rmse, r2, mape, accuracy]
})
eval_df.to_csv("supervised_evaluation.csv", index=False)
print("✅ Metrik evaluasi disimpan ke 'model/supervised_evaluation.csv'")



# ========== 8. Feature Importance ==========
onehot_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical)
all_features = numerical + list(onehot_features)
importances = model.named_steps['regressor'].feature_importances_
feat_importance = pd.Series(importances, index=all_features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance dari Random Forest")
plt.xlabel("Importance")
plt.ylabel("Fitur")
plt.tight_layout()
plt.savefig("feature_importance_supervised.png")
plt.show()

# ========== 9. Simpan Model ==========
joblib.dump(model, "model_prediksi_rent.pkl")
print("\n✅ Model berhasil disimpan ke file 'model_prediksi_rent.pkl'")
