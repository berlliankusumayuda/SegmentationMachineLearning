import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🎯 Mall Customer Segmentation")
st.markdown("""
Analisis segmentasi pelanggan mall berdasarkan perilaku belanja menggunakan *K-Means* dan *Hierarchical Clustering*.
""")

uploaded_file = st.file_uploader("📂 Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Asli")
    st.dataframe(df, use_container_width=True)

    # Preprocessing
    if 'CustomerID' in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)
    if 'Gender' in df.columns:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("⚙ Pilih Fitur untuk Clustering")
    selected_features = st.multiselect("Pilih fitur yang digunakan untuk clustering", numeric_cols, default=numeric_cols)

    if selected_features:
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Pilihan Algoritma
        algo = st.radio("Pilih Algoritma Clustering", ["K-Means", "Hierarchical Clustering"], horizontal=True)
        n_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=4)

        if algo == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(X_scaled)
            inertia_value = model.inertia_
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = model.fit_predict(X_scaled)
            inertia_value = None  # Tidak tersedia untuk Hierarchical

        df['Cluster'] = cluster_labels

        # Evaluasi Clustering
        sil_score = silhouette_score(X_scaled, cluster_labels)
        st.subheader("📈 Evaluasi Clustering")
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        if inertia_value is not None:
            st.metric("Inertia (SSE)", f"{inertia_value:.2f}")

        # Visualisasi 2D
        st.subheader("📊 Visualisasi Scatter 2D")
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=cluster_labels, palette='tab10', ax=ax)
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        st.pyplot(fig)

        # Dendrogram untuk Hierarchical
        if algo == "Hierarchical Clustering":
            st.subheader("🌳 Dendrogram (Hierarchical Clustering)")
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
            plt.xlabel("Data Points")
            plt.ylabel("Distance")
            st.pyplot(fig)

        # Elbow Method
        if algo == "K-Means":
            st.subheader("📈 Elbow Method (Untuk K-Means)")
            distortions = []
            K_range = range(1, 11)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                distortions.append(km.inertia_)

            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(K_range, distortions, 'bo-')
            ax_elbow.set_xlabel('Jumlah Cluster (k)')
            ax_elbow.set_ylabel('Inertia (SSE)')
            ax_elbow.set_title('Elbow Method')
            st.pyplot(fig_elbow)

            # Download hasil clustering
            st.subheader("⬇ Download Hasil Clustering")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, file_name="hasil_clustering.csv", mime="text/csv")

else:
    st.info("Silakan upload dataset .csv terlebih dahulu.")
