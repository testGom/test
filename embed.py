@st.cache_data(show_spinner=True)
def reduce_embeddings(embeddings):
    embeddings = np.array(embeddings)
    tqdm.write(f"Reducing embeddings with UMAP... shape={embeddings.shape}")
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    tqdm.write("UMAP reduction complete.")
    return reducer, reduced

def format_text_multiline(text, max_chars=60, max_lines=10):
    words = text.split()
    lines, line = [], ""
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

def embed_with_ollama(text: str, model: str = "nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    response = requests.post(url, json={"model": model, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

def visualize3d(collection):
    data = collection.get(include=["embeddings", "documents"])
    embeddings = np.array(data.get("embeddings", []))
    ids = data.get("ids", [])
    documents = data.get("documents", [])

    if len(embeddings) == 0:
        st.warning("No embeddings found in this collection.")
        return

    # Dimensionality reduction
    reducer, reduced = reduce_embeddings(embeddings)

    # Sidebar clustering controls
    st.sidebar.markdown("## Clustering Controls")
    n_clusters = st.sidebar.slider("Number of clusters", 2, 20, 5)

    if st.sidebar.button("Cluster"):
        with st.spinner("Clustering with KMeans..."):
            tqdm.write(f"Clustering with KMeans: k={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(reduced)
            tqdm.write("Clustering complete.")
    else:
        labels = [-1] * len(reduced)
        kmeans = None  # placeholder to prevent usage without clustering

    # Prepare DataFrame
    df = pd.DataFrame(reduced, columns=["x", "y", "z"])
    df["id"] = ids
    df["doc"] = [format_text_multiline(d) for d in documents]
    df["label"] = labels
    df["x_str"] = df["x"].round(3).astype(str)
    df["y_str"] = df["y"].round(3).astype(str)
    df["z_str"] = df["z"].round(3).astype(str)

    fig = px.scatter_3d(df, x="x", y="y", z="z", color=df["label"].astype(str))
    fig.update_traces(
        hovertemplate="""
            <b>ID:</b> %{customdata[0]}<br>
            <b>Text:</b><br>%{customdata[1]}<br><br>
            <b>X:</b> %{customdata[2]}<br>
            <b>Y:</b> %{customdata[3]}<br>
            <b>Z:</b> %{customdata[4]}<extra></extra>
        """,
        customdata=np.stack((df["id"], df["doc"], df["x_str"], df["y_str"], df["z_str"]), axis=-1)
    )

    st.markdown("### 3D Embedding Visualization")
    st.plotly_chart(fig, use_container_width=True)

    # --- Query input ---
    st.markdown("---")
    st.markdown("### 🔍 Try a new query (not saved)")
    user_input = st.text_input("Enter a text query:")

    if user_input and kmeans:
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
                    hovertemplate="""
                        <b>ID:</b> %{customdata[0]}<br>
                        <b>Text:</b><br>%{customdata[1]}<br><br>
                        <b>X:</b> %{customdata[2]}<br>
                        <b>Y:</b> %{customdata[3]}<br>
                        <b>Z:</b> %{customdata[4]}<extra></extra>
                    """,
                    customdata=np.array([["your input", format_text_multiline(user_input),
                                          f"{xq:.3f}", f"{yq:.3f}", f"{zq:.3f}"]])
                ).data[0])

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error embedding text: {e}")
