"""
face_similarity.py
-------------------
Core PCA-based face similarity logic, shared by the notebook and the
Streamlit app so both use the identical pipeline.

Dataset: the Olivetti Faces dataset (400 grayscale 64x64 images, 40 people,
10 images each), loaded via scikit-learn's built-in fetcher. On first run
this needs a one-time internet connection to download ~4MB; scikit-learn
then caches it locally under ~/scikit_learn_data/ for every run after that.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

IMAGE_SHAPE = (64, 64)


def load_faces(shuffle: bool = False, random_state: int = 42):
    """
    Returns
    -------
    images : (400, 64, 64) float32 array, pixel values in [0, 1]
    data   : (400, 4096) float32 array -- the same images, flattened
    target : (400,) int array -- person ID, 0-39 (10 images per person)
    """
    bunch = fetch_olivetti_faces(shuffle=shuffle, random_state=random_state)
    return bunch.images, bunch.data, bunch.target


def fit_eigenfaces(X: np.ndarray, n_components: float | int = 0.95, whiten: bool = True) -> PCA:
    """
    Fit PCA ("eigenfaces") on a set of flattened face vectors.
    n_components=0.95 keeps enough components to explain 95% of variance,
    which is the standard eigenfaces setup -- compresses 4096 raw pixels
    down to a much smaller, more meaningful representation.
    """
    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    pca.fit(X)
    return pca


def project(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Project raw flattened face vectors into eigenface space."""
    return pca.transform(X)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def all_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "euclidean": euclidean_distance(a, b),
        "cosine": cosine_sim(a, b),
        "manhattan": manhattan_distance(a, b),
    }


def rank_by_metric(query_vec: np.ndarray, gallery_vecs: np.ndarray, metric: str, top_n: int):
    """
    Rank every gallery vector against the query by the chosen metric.
    euclidean/manhattan: smaller = more similar (ascending sort)
    cosine: larger = more similar (descending sort)
    Returns a list of dicts: {index, euclidean, cosine, manhattan}
    """
    scored = []
    for i, vec in enumerate(gallery_vecs):
        m = all_metrics(query_vec, vec)
        scored.append({"index": i, **m})

    reverse = metric == "cosine"
    scored.sort(key=lambda r: r[metric], reverse=reverse)
    return scored[:top_n]


def leave_one_out_accuracy(embeddings: np.ndarray, target: np.ndarray, metric: str, k: int = 1) -> float:
    """
    For every face, treat it as a query against every OTHER face and check
    whether the top-k nearest matches (by the given metric) include at
    least one image of the same person. This is the standard way to
    evaluate a similarity/retrieval system without needing separate labels
    at inference time -- the labels here are only used for scoring, never
    fed into PCA or the distance computation itself.

    Vectorized: builds one full pairwise-distance/similarity matrix instead
    of looping over every (query, candidate) pair in Python.
    """
    from scipy.spatial.distance import cdist

    n = len(embeddings)
    if metric == "cosine":
        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, -np.inf)  # never match a face to itself
        order = np.argsort(-sim, axis=1)  # descending: most similar first
    else:
        dist_metric = "euclidean" if metric == "euclidean" else "cityblock"
        dist = cdist(embeddings, embeddings, metric=dist_metric)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)  # ascending: closest first

    top_k = order[:, :k]
    hits = sum(target[i] in target[top_k[i]] for i in range(n))
    return hits / n
