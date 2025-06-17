import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import umap
import hdbscan
from chromadb import PersistentClient
from sklearn.cluster import KMeans
import requests

def embed_with_ollama(text: str, model: str = "nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    response = requests.post(url, json={"model": model, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]


# Sidebar clustering params (no compiler required)
st.sidebar.markdown("## Clustering Controls")
n_clusters = st.sidebar.slider("Number of clusters", 2, 20, 5)

# Cluster on button press
if st.sidebar.button("Cluster"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)
else:
    labels = [-1] * len(reduced)  # No clusters
df["label"] = labels
fig = px.scatter_3d(df, x="x", y="y", z="z", color=df["label"].astype(str))

# Load ChromaDB data
client = PersistentClient(path="./chroma_db")
collection = client.get_collection("AA")
data = collection.get(include=["embeddings", "documents"])

embeddings = np.array(data["embeddings"])
ids = data["ids"]
docs = data["documents"]

# UMAP reduction
reducer = umap.UMAP(n_components=3, random_state=42)
reduced = reducer.fit_transform(embeddings)

# Sidebar clustering parameters
st.sidebar.markdown("## Clustering Controls")
min_cluster_size = st.sidebar.slider("Min cluster size", 2, 100, 15)
min_samples = st.sidebar.slider("Min samples", 1, 20, 5)

# Cluster on button press
if st.sidebar.button("Cluster"):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(reduced)
else:
    labels = [-1] * len(reduced)  # no cluster

# Prepare DataFrame
def format_text_multiline(text, max_chars=60, max_lines=10):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line += " " + word if line else word
        else:
            lines.append(line)
            line = word
        if len(lines) >= max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    return "<br>".join(lines)

df = pd.DataFrame(reduced, columns=["x", "y", "z"])
df["id"] = ids
df["doc"] = [format_text_multiline(d) for d in docs]
df["label"] = labels

# Hover-friendly fields
df["x_str"] = df["x"].round(3).astype(str)
df["y_str"] = df["y"].round(3).astype(str)
df["z_str"] = df["z"].round(3).astype(str)

# 3D Plot
fig = px.scatter_3d(df, x="x", y="y", z="z", color=df["label"].astype(str))

fig.update_traces(
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Text:</b><br>%{customdata[1]}<br><br>" +
        "<b>X:</b> %{customdata[2]}<br>" +
        "<b>Y:</b> %{customdata[3]}<br>" +
        "<b>Z:</b> %{customdata[4]}<extra></extra>",
    customdata=np.stack((df["id"], df["doc"], df["x_str"], df["y_str"], df["z_str"]), axis=-1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🔍 Try a new query (not saved)")

user_input = st.text_input("Enter a text query:")

if user_input:
    with st.spinner("Embedding and projecting input..."):
        try:
            query_embedding = np.array([embed_with_ollama(user_input)])
            reduced_query = reducer.transform(query_embedding)
            predicted_label = kmeans.predict(reduced_query)[0]

            xq, yq, zq = reduced_query[0]

            query_df = pd.DataFrame({
                "x": [xq],
                "y": [yq],
                "z": [zq],
                "label": [f"input → {predicted_label}"],
                "id": ["your input"],
                "doc": [format_text_multiline(user_input)],
                "x_str": [f"{xq:.3f}"],
                "y_str": [f"{yq:.3f}"],
                "z_str": [f"{zq:.3f}"]
            })

            fig.add_trace(px.scatter_3d(
                query_df, x="x", y="y", z="z",
                color_discrete_sequence=["black"]
            ).update_traces(
                marker=dict(size=8, symbol="diamond"),
                hovertemplate=
                    "<b>ID:</b> %{customdata[0]}<br>" +
                    "<b>Text:</b><br>%{customdata[1]}<br><br>" +
                    "<b>X:</b> %{customdata[2]}<br>" +
                    "<b>Y:</b> %{customdata[3]}<br>" +
                    "<b>Z:</b> %{customdata[4]}<extra></extra>",
                customdata=np.array([[query_df["id"][0], query_df["doc"][0],
                                      query_df["x_str"][0], query_df["y_str"][0], query_df["z_str"][0]]])
            ).data[0])

        except Exception as e:
            st.error(f"Error embedding text: {e}")
