"""
precompute_cache.py
--------------------
Run this ONCE, locally, with a normal internet connection, before you
deploy the app:

    python scripts/precompute_cache.py

It downloads the Olivetti Faces dataset (via sklearn's built-in fetcher),
fits the eigenfaces PCA, and saves everything the app needs into a single
small file: models/face_cache.joblib (~9 MB).

Commit that file to your repo. From then on, app/app.py loads straight
from the cache instead of re-downloading the dataset and re-fitting PCA
every time the app cold-starts (e.g. on Streamlit Community Cloud, which
spins up a fresh container after periods of inactivity) -- so the app
opens instantly instead of pausing on first load.

If models/face_cache.joblib is missing (e.g. you skip this step), the app
falls back to fetching + fitting live, exactly as before -- nothing breaks,
it's just slower on cold starts.
"""

import os
import sys

import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from face_similarity import load_faces, fit_eigenfaces, project  # noqa: E402

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_cache.joblib")


def main():
    print("Fetching Olivetti Faces dataset (downloads once, ~4MB)...")
    images, data, target = load_faces(shuffle=False)

    print("Fitting eigenfaces (PCA, 95% variance)...")
    pca = fit_eigenfaces(data, n_components=0.95, whiten=True)
    embeddings = project(pca, data)

    print(f"Eigenfaces kept: {pca.n_components_}  (compressed from {data.shape[1]}D)")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    joblib.dump(
        {
            "images": images,       # (400, 64, 64) -- kept for display in the app
            "target": target,       # (400,) person IDs
            "pca": pca,             # fitted PCA model -- used to project new query photos too
            "embeddings": embeddings,  # (400, n_components) -- precomputed gallery embeddings
        },
        CACHE_PATH,
        compress=3,
    )

    size_mb = os.path.getsize(CACHE_PATH) / (1024 * 1024)
    print(f"\nSaved cache to {os.path.abspath(CACHE_PATH)} ({size_mb:.1f} MB)")
    print("Commit this file to your repo -- app/app.py will load it instantly from now on.")


if __name__ == "__main__":
    main()
