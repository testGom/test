# chroma_3d_viewer.py

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
import pyqtgraph.opengl as gl
from chromadb import PersistentClient
import umap
from tqdm import tqdm

class Scatter3DApp(QMainWindow):
    def __init__(self, embeddings, ids, texts):
        super().__init__()
        self.embeddings = embeddings
        self.ids = ids
        self.texts = texts
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('ChromaDB 3D Embedding Viewer')
        self.resize(1000, 800)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.opts['distance'] = 10
        self.gl_widget.setCameraPosition(elevation=15, azimuth=45)
        self.gl_widget.mousePressEvent = self.on_mouse_press

        pos = np.array(self.embeddings)
        pos -= pos.mean(axis=0)
        pos /= pos.std()

        self.scatter = gl.GLScatterPlotItem(pos=pos, size=0.1, color=(0, 0.5, 1, 1), pxMode=False)
        self.gl_widget.addItem(self.scatter)

        self.label = QLabel("Click a point to see document text")
        self.label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.gl_widget)
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_mouse_press(self, event):
        cam_pos = self.gl_widget.cameraPosition()
        dists = np.linalg.norm(self.embeddings - cam_pos, axis=1)
        nearest_idx = np.argmin(dists)
        self.label.setText(f"**ID**: {self.ids[nearest_idx]}\n\n**Document**:\n{self.texts[nearest_idx]}")

def reduce_embeddings(embeddings):
    embeddings = np.array(embeddings)
    tqdm.write(f"Starting dim red on shape {embeddings.shape}")
    reducer = umap.UMAP(n_components=3, random_state=42, verbose=True)
    reduced = reducer.fit_transform(embeddings)
    tqdm.write("Reduction complete")
    return reduced

if __name__ == '__main__':
    # Load from ChromaDB
    client = PersistentClient(path="./chroma_db")
    collection = client.get_collection("AA")
    data = collection.get(include=["embeddings", "documents"])

    ids = data.get("ids", [])
    embeddings = data.get("embeddings", [])
    documents = data.get("documents", [])

    reduced = reduce_embeddings(embeddings)

    app = QApplication(sys.argv)
    viewer = Scatter3DApp(reduced, ids, documents)
    viewer.show()
    sys.exit(app.exec_())
