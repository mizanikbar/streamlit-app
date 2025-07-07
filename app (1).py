
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# --- Sidebar Navigation ---
st.sidebar.title("Main Page")
page = st.sidebar.radio("Navigation", ["🏠 Main Page", "📊 Classification", "📈 Clustering"])

# --- Main Page ---
if page == "🏠 Main Page":
    st.title("Ujian Akhir Semester")
    st.subheader("Streamlit Apps")
    st.markdown("Collection of my apps deployed in Streamlit")
    st.markdown("**Nama:** Mizan Ikbar")
    st.markdown("**NIM:** 22146003")

# --- Classification Page ---
elif page == "📈 Clustering":
    st.title("Clustering Pelanggan Menggunakan Gender, Age, Income, dan Spend Score")
    st.write("Proyek ini menggunakan K-Means untuk mengelompokkan pelanggan berdasarkan gender, usia, pendapatan, dan skor belanja.")

    data = pd.read_csv("lokasi_gerai_kopi_clean.csv")
    st.write("### Data Sample")
    st.dataframe(data.head())

    # Encode gender (misal: Male = 1, Female = 0)
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder()
    data["gender_encoded"] = le.fit_transform(data["gender"])

    # Fitur yang digunakan
    X = data[["gender_encoded", "age", "income", "spend_score"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    data["Cluster"] = clusters

    # Visualisasi dengan PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data["PCA1"] = X_pca[:, 0]
    data["PCA2"] = X_pca[:, 1]

    st.write("### Visualisasi Clustering (PCA 2D)")
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="tab10", data=data, ax=ax)
    st.pyplot(fig)

    # Input data baru
    st.write("### Prediksi Cluster Pelanggan Baru")
    gender_new = st.selectbox("Pilih Gender", ["Female", "Male"])
    gender_encoded = 0 if gender_new == "Female" else 1
    age_new = st.number_input("Masukkan usia (age)", value=float(data["age"].mean()))
    income_new = st.number_input("Masukkan pendapatan (income)", value=float(data["income"].mean()))
    score_new = st.number_input("Masukkan skor belanja (spend_score)", value=float(data["spend_score"].mean()))

    if st.button("Prediksi Cluster"):
        new_scaled = scaler.transform([[gender_encoded, age_new, income_new, score_new]])
        new_cluster = kmeans.predict(new_scaled)[0]
        st.success(f"Pelanggan baru masuk ke dalam Cluster {new_cluster}")
