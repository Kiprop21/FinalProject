import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.title("Clustering Project")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Perform clustering
    n_clusters=st.slider("Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)
    df['cluster'] = kmeans.labels_

    # Visualize
    st.subheader(f"Clusters with {n_clusters} Centers")
    fig, ax =plt.subplots()
    sns.scatterplot(x='lon_scaled', y='lat_scaled', hue='cluster', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)



   