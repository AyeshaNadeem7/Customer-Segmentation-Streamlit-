import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")  # Ensure the file is in same folder
    return df

df = load_data()

# Page Title
st.title("Mall Customer Segmentation")
st.write("This app performs customer segmentation using **K-Means** and **Hierarchical Clustering**.")

# Preview Data
st.subheader("Dataset Preview")
st.dataframe(df.head())

# EDA
st.subheader("Exploratory Data Analysis")
st.write(df.describe())

# Gender distribution
fig, ax = plt.subplots()
sns.countplot(x="Gender", data=df, ax=ax)
ax.set_title("Gender Distribution")
st.pyplot(fig)

# Feature Scaling
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
st.header("K-Means Clustering")
k_clusters = st.sidebar.slider("Number of clusters for K-Means", 2, 10, 4)
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
sil_kmeans = silhouette_score(X_scaled, df["Cluster_KMeans"])
st.write(f"**K-Means Silhouette Score:** {sil_kmeans:.3f}")

# 2D Plot K-Means
fig_km, ax = plt.subplots()
sns.scatterplot(
    x=X_scaled[:, 1],  # Annual Income
    y=X_scaled[:, 2],  # Spending Score
    hue=df["Cluster_KMeans"],
    palette="tab10",
    ax=ax
)
ax.set_title("K-Means Customer Segments")
handles, labels_plot = ax.get_legend_handles_labels()
labels_plot = [f"Cluster {label}" for label in labels_plot]
ax.legend(handles, labels_plot)
st.pyplot(fig_km)

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig_elbow, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title("Elbow Method (K-Means)")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("WCSS")
st.pyplot(fig_elbow)

# 3D Plot
if st.sidebar.checkbox("Show 3D Plot for K-Means"):
    from mpl_toolkits.mplot3d import Axes3D
    fig_3d_k = plt.figure()
    ax3d_k = fig_3d_k.add_subplot(111, projection="3d")
    scatter = ax3d_k.scatter(
        df["Age"], df["Annual Income (k$)"], df["Spending Score (1-100)"],
        c=df["Cluster_KMeans"], cmap="tab10", s=50
    )
    ax3d_k.set_xlabel("Age")
    ax3d_k.set_ylabel("Annual Income (k$)")
    ax3d_k.set_zlabel("Spending Score (1-100)")
    ax3d_k.set_title("K-Means Clusters (3D)")
    legend_labels = [f"Cluster {int(label)}" for label in np.unique(df["Cluster_KMeans"])]
    ax3d_k.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    st.pyplot(fig_3d_k)

# Hierarchical Clustering 
st.header("Hierarchical Clustering")
n_clusters = st.sidebar.slider("Number of clusters for Hierarchical", 2, 10, 4)
linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])

# Fit Hierarchical
hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='euclidean',  # Updated for sklearn >= 1.2
    linkage=linkage_method
)
df["Cluster_HC"] = hc.fit_predict(X_scaled)
sil_hc = silhouette_score(X_scaled, df["Cluster_HC"])
st.write(f"**Hierarchical Silhouette Score:** {sil_hc:.3f}")

# 2D Plot Hierarchical
fig_hc, ax = plt.subplots()
sns.scatterplot(
    x=X_scaled[:, 1],
    y=X_scaled[:, 2],
    hue=df["Cluster_HC"],
    palette="tab10",
    ax=ax
)
ax.set_title("Hierarchical Customer Segments")
handles, labels_plot = ax.get_legend_handles_labels()
labels_plot = [f"Cluster {label}" for label in labels_plot]
ax.legend(handles, labels_plot)
st.pyplot(fig_hc)

# Dendrogram
st.subheader("Dendrogram")
linked = linkage(X_scaled, method=linkage_method, metric="euclidean")
fig_dend, ax = plt.subplots(figsize=(10, 5))
dendrogram(linked, orientation="top", distance_sort="descending", show_leaf_counts=False)
st.pyplot(fig_dend)

# Show Clustered Data
st.subheader("Clustered Data")
st.write(df.head())

# Business Insights
st.header("Business Insights")
st.markdown("""
- **Cluster 0**: High income, high spenders → Target for premium offers.
- **Cluster 1**: High income, low spenders → Upselling opportunities.
- **Cluster 2**: Low income, high spenders → Price-sensitive but loyal customers.
- **Cluster 3**: Low income, low spenders → Less profitable segment.
""")
