# ===== 1. Import Library =====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

# ===== 2. Load Dataset =====
df = pd.read_csv("House_Rent_Dataset.csv")

# ===== 3. Data Understanding =====
print("==== Informasi Dataset ====")
print(df.info())
print("\n==== Statistik Deskriptif ====")
print(df.describe())
print("\n==== 5 Data Teratas ====")
print(df.head())

# Drop kolom yang tidak relevan
df.drop(columns=['Posted On'], inplace=True)

# ===== 4. Visualisasi Sebelum Pembersihan =====
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Rent', 'Size', 'BHK', 'Bathroom']])
plt.title("Boxplot Sebelum Pembersihan")
plt.savefig("uns_boxplot_before_cleaning.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df[['Rent', 'Size', 'BHK', 'Bathroom']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Sebelum Pembersihan")
plt.savefig("uns_correlation_matrix_before.png")
plt.show()

# ===== 5. Data Cleansing =====
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

# ===== 6. Visualisasi Setelah Pembersihan =====
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Rent', 'Size', 'BHK', 'Bathroom']])
plt.title("Boxplot Setelah Pembersihan")
plt.savefig("uns_boxplot_after_cleaning.png")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df[['Rent', 'Size', 'BHK', 'Bathroom']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Setelah Pembersihan")
plt.savefig("uns_correlation_matrix_after.png")
plt.show()

# ===== 7. Distribusi Fitur Numerik =====
numerical_features = ['Rent', 'Size', 'BHK', 'Bathroom']
for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=True, color='lightgreen', edgecolor='black', linewidth=0.5)
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"uns_dist_{col.lower()}.png")
    plt.show()

# ===== 8. Persiapan Fitur untuk Clustering =====
clustering_df = df[['Size', 'BHK', 'Bathroom', 'Rent']]

# Normalisasi data
scaler = StandardScaler()
clustering_scaled = scaler.fit_transform(clustering_df)

# ===== 9. Elbow Method untuk Menentukan k =====
sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(clustering_scaled)
    sse.append(km.inertia_)

plt.plot(range(1, 10), sse, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.grid(True)
plt.tight_layout()
plt.savefig("uns_elbow_method.png")
plt.show()

# ===== 10. Clustering dengan k = 4 (contoh) =====
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(clustering_scaled)
df['Cluster'] = labels

# ===== 11. Visualisasi Cluster dengan PCA =====
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(clustering_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=labels, palette='Set2')
plt.title("Visualisasi Cluster (PCA)")
plt.xlabel("PCA Komponen 1")
plt.ylabel("PCA Komponen 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("uns_pca_clusters.png")
plt.show()

# ===== 12. Evaluasi dan Interpretasi Cluster =====
print("\n=== Statistik Rata-rata per Cluster ===")
cluster_summary = df.groupby('Cluster')[['Rent', 'Size', 'BHK', 'Bathroom']].mean().round(2)
print(cluster_summary)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Cluster', y='Rent', palette='Set2')
plt.title("Distribusi Rent per Cluster")
plt.savefig("uns_rent_per_cluster.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Cluster', palette='Set3')
plt.title("Jumlah Properti per Cluster")
plt.savefig("uns_count_per_cluster.png")
plt.show()

# ===== 13. Simpan Hasil Clustering ke CSV =====
cluster_summary.to_csv("cluster_summary.csv")
print("\n✅ Data ringkasan cluster berhasil disimpan ke 'cluster_summary.csv'")

# Simpan komponen PCA ke dataframe
df['PCA1'] = reduced_data[:, 0]
df['PCA2'] = reduced_data[:, 1]

# Simpan ke CSV
df.to_csv("clustered_data.csv", index=False)


