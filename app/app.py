import io
import os
import sys
import csv

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from face_similarity import (  # noqa: E402
    load_faces, fit_eigenfaces, project, rank_by_metric, IMAGE_SHAPE
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EigenMatch",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (same visual language as PixelMatch) ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main .block-container { padding: 2rem 3rem 4rem; max-width: 1280px; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #0c0e14; color: #e8eaf0; }

.hero { display: flex; align-items: center; gap: 1rem; padding: 2.5rem 0 1rem; }
.hero-icon {
    font-size: 2.6rem; width: 64px; height: 64px;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    border-radius: 16px; display: grid; place-items: center;
    box-shadow: 0 0 32px #6c63ff55;
}
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.03em; color: #fff; line-height: 1; margin: 0; }
.hero-sub { font-size: 0.95rem; color: #7c8099; margin-top: 0.3rem; }

.card { background: #13151f; border: 1px solid #1e2130; border-radius: 16px; padding: 1.6rem; }

.section-label {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; color: #6c63ff; margin-bottom: 0.6rem;
}

.pill {
    display: inline-block; background: #1a1d2e; border: 1px solid #2a2d40;
    border-radius: 999px; padding: 0.2rem 0.75rem; font-size: 0.78rem; color: #a0a3b1;
}
.pill-accent { background: #1f1a3a; border-color: #6c63ff44; color: #a78bfa; }

[data-testid="stFileUploader"] {
    background: #13151f !important; border: 1.5px dashed #2a2d40 !important;
    border-radius: 12px !important; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #6c63ff !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important; padding: 0.65rem 2rem !important;
    box-shadow: 0 4px 20px #6c63ff44 !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important; box-shadow: 0 6px 28px #6c63ff66 !important;
}

[data-testid="stMetric"] {
    background: #0f1118; border: 1px solid #1e2130; border-radius: 10px; padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #7c8099 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'Syne', sans-serif !important; }

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6c63ff, #a78bfa) !important; border-radius: 999px !important;
}

[data-testid="stSidebar"] { background: #0f1118 !important; border-right: 1px solid #1e2130; }
[data-testid="stExpander"] { background: #13151f !important; border: 1px solid #1e2130 !important; border-radius: 12px !important; }
hr { border-color: #1e2130 !important; }
[data-testid="stAlert"] { background: #13151f !important; border: 1px solid #1e2130 !important; border-radius: 12px !important; color: #a0a3b1 !important; }
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_cache.joblib")


@st.cache_resource(show_spinner="Loading eigenfaces gallery...")
def get_gallery():
    """
    Loads instantly from models/face_cache.joblib if it exists (generated
    once via `python scripts/precompute_cache.py`, then committed to the
    repo). Otherwise falls back to fetching the dataset live and fitting
    PCA on the spot -- slower, but nothing breaks if the cache is missing.
    """
    if os.path.exists(CACHE_PATH):
        import joblib
        cached = joblib.load(CACHE_PATH)
        images, target = cached["images"], cached["target"]
        pca, embeddings = cached["pca"], cached["embeddings"]
        data = images.reshape(len(images), -1)
        return images, data, target, pca, embeddings

    images, data, target = load_faces(shuffle=False)
    pca = fit_eigenfaces(data, n_components=0.95, whiten=True)
    embeddings = project(pca, data)
    return images, data, target, pca, embeddings


def preprocess_uploaded_face(img: Image.Image) -> np.ndarray:
    """Match an uploaded photo to the Olivetti preprocessing: grayscale, 64x64, [0,1]."""
    img = img.convert("L").resize(IMAGE_SHAPE[::-1])
    vec = np.array(img, dtype=np.float32).flatten() / 255.0
    return vec


def results_to_csv(results: list) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["rank", "label", "person", "euclidean", "cosine", "manhattan"])
    writer.writeheader()
    for rank, r in enumerate(results, 1):
        writer.writerow({
            "rank": rank, "label": r["label"], "person": r.get("person", ""),
            **{k: f"{r[k]:.6f}" for k in ["euclidean", "cosine", "manhattan"]},
        })
    return buf.getvalue().encode()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">✦</div>
  <div>
    <div class="hero-title">EigenMatch</div>
    <div class="hero-sub">PCA-based face similarity search — Euclidean, Cosine &amp; Manhattan, side by side</div>
  </div>
</div>
""", unsafe_allow_html=True)

images, data, target, pca, embeddings = get_gallery()

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
    top_n = st.slider("Top-N results", 1, 15, 5)
    metric = st.selectbox("Ranking metric", ["euclidean", "cosine", "manhattan"],
                           help="euclidean ↓ better | cosine ↑ better | manhattan ↓ better")
    st.markdown(f'<span class="pill pill-accent">✦ {pca.n_components_} eigenfaces (95% variance)</span>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Metric guide</div>', unsafe_allow_html=True)
    st.markdown("""
<span style="color:#a0a3b1;font-size:0.85rem">
**Euclidean ↓** — straight-line distance in eigenface space<br><br>
**Cosine ↑** — angle between eigenface vectors, ignores overall brightness<br><br>
**Manhattan ↓** — sum of absolute differences, robust to outlier dimensions
</span>
""", unsafe_allow_html=True)

# ── Query selection ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">① Choose your faces</div>', unsafe_allow_html=True)

mode = st.radio(
    "Query source",
    [
        "Pick from the gallery",
        "Upload one photo (vs. built-in gallery)",
        "Upload multiple photos (compare them with each other)",
    ],
    horizontal=True, label_visibility="collapsed",
)

# gallery_items: list of dicts the rest of the app works with, regardless of mode
# {"label": str, "image": np.ndarray (64x64), "vec": np.ndarray (eigenface space), "person": int|None}
query_item = None
gallery_items = []

if mode == "Pick from the gallery":
    col_a, col_b = st.columns([1, 3])
    with col_a:
        person_id = st.selectbox("Person (0-39)", sorted(set(target)))
    with col_b:
        indices_for_person = [i for i in range(len(target)) if target[i] == person_id]
        thumb_cols = st.columns(len(indices_for_person))
        for i, idx in enumerate(indices_for_person):
            with thumb_cols[i]:
                st.image(images[idx], width="stretch", caption=f"#{idx}")
        chosen_idx = st.select_slider("Image index", options=indices_for_person, value=indices_for_person[0])

    query_item = {"label": f"gallery face #{chosen_idx} (person {person_id})",
                  "image": images[chosen_idx], "vec": embeddings[chosen_idx], "person": person_id}
    gallery_items = [
        {"label": f"face #{i} (person {int(target[i])})", "image": images[i], "vec": embeddings[i], "person": int(target[i])}
        for i in range(len(embeddings)) if i != chosen_idx
    ]

elif mode == "Upload one photo (vs. built-in gallery)":
    uploaded = st.file_uploader("Grayscale (or color, auto-converted) face photo, ideally cropped to just the face",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        raw_vec = preprocess_uploaded_face(Image.open(uploaded))
        query_item = {"label": uploaded.name, "image": raw_vec.reshape(IMAGE_SHAPE),
                      "vec": pca.transform(raw_vec.reshape(1, -1))[0], "person": None}
        gallery_items = [
            {"label": f"face #{i} (person {int(target[i])})", "image": images[i], "vec": embeddings[i], "person": int(target[i])}
            for i in range(len(embeddings))
        ]

else:  # Upload multiple photos, compare with each other
    uploaded_files = st.file_uploader(
        "Upload 2 or more face photos to compare against each other",
        type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True,
    )
    if uploaded_files and len(uploaded_files) < 2:
        st.warning("Upload at least 2 photos so there's something to compare against.")
    elif uploaded_files and len(uploaded_files) >= 2:
        all_items = []
        for f in uploaded_files:
            raw_vec = preprocess_uploaded_face(Image.open(f))
            all_items.append({
                "label": f.name, "image": raw_vec.reshape(IMAGE_SHAPE),
                "vec": pca.transform(raw_vec.reshape(1, -1))[0], "person": None,
            })

        st.markdown("<br>", unsafe_allow_html=True)
        thumb_cols = st.columns(min(len(all_items), 6))
        for i, item in enumerate(all_items):
            with thumb_cols[i % len(thumb_cols)]:
                st.image(item["image"], width="stretch", caption=item["label"])

        query_name = st.selectbox("Which photo should be the query?", [it["label"] for it in all_items])
        query_item = next(it for it in all_items if it["label"] == query_name)
        gallery_items = [it for it in all_items if it["label"] != query_name]

if query_item is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    qc1, qc2 = st.columns([1, 4])
    with qc1:
        st.image(query_item["image"], width="stretch")
    with qc2:
        st.markdown(f'<span class="pill">{query_item["label"]}</span>', unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
ready = query_item is not None and len(gallery_items) > 0
run = st.button("✦ Find Similar Faces", type="primary", disabled=not ready)

if not ready and not run:
    st.info("Pick a face from the gallery, or upload photo(s), to get started.")

if run and ready:
    with st.spinner("Ranking by similarity..."):
        gallery_vecs = np.array([it["vec"] for it in gallery_items])
        capped_top_n = min(top_n, len(gallery_items))
        results = rank_by_metric(query_item["vec"], gallery_vecs, metric, top_n=capped_top_n)
        for r in results:
            matched_item = gallery_items[r["index"]]
            r["label"] = matched_item["label"]
            r["image"] = matched_item["image"]
            r["person"] = matched_item["person"]

    st.markdown("<hr>", unsafe_allow_html=True)
    hdr_col1, hdr_col2 = st.columns([3, 1])
    with hdr_col1:
        st.markdown(f'<div class="section-label">Results — top {len(results)} matches (ranked by {metric})</div>',
                    unsafe_allow_html=True)
        if query_item["person"] is not None:
            n_correct = sum(1 for r in results if r["person"] == query_item["person"])
            st.markdown(f'<span class="pill pill-accent">{n_correct}/{len(results)} are the same person</span>',
                        unsafe_allow_html=True)
    with hdr_col2:
        st.download_button("⬇ Export CSV", data=results_to_csv(results),
                            file_name="face_similarity_results.csv", mime="text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    for rank, r in enumerate(results, 1):
        euc_norm = 1.0 / (1.0 + r["euclidean"])
        cos_pct = max(0.0, min(1.0, r["cosine"]))
        man_norm = 1.0 / (1.0 + r["manhattan"])

        match_tag = ""
        if query_item["person"] is not None and r["person"] is not None:
            match_tag = "  ✓ same person" if r["person"] == query_item["person"] else "  ✗ different person"

        with st.expander(f"  #{rank}  —  {r['label']}{match_tag}", expanded=(rank <= 3)):
            ic, mc = st.columns([1, 3], gap="large")
            with ic:
                st.image(r["image"], width="stretch")
            with mc:
                m1, m2, m3 = st.columns(3)
                m1.metric("Euclidean", f"{r['euclidean']:.4f}", help="↓ lower = more similar")
                m2.metric("Cosine", f"{r['cosine']:.4f}", help="↑ higher = more similar")
                m3.metric("Manhattan", f"{r['manhattan']:.4f}", help="↓ lower = more similar")
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(euc_norm, text=f"Euclidean closeness: {euc_norm*100:.1f}%")
                st.progress(cos_pct, text=f"Cosine match: {cos_pct*100:.1f}%")
                st.progress(man_norm, text=f"Manhattan closeness: {man_norm*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Full Results Table</div>', unsafe_allow_html=True)
    df = pd.DataFrame([{k: r[k] for k in ["label", "person", "euclidean", "cosine", "manhattan"]} for r in results])
    df.insert(0, "rank", range(1, len(df) + 1))
    df.columns = ["Rank", "Image", "Person", "Euclidean ↓", "Cosine ↑", "Manhattan ↓"]
    st.dataframe(
        df.style
          .highlight_min("Euclidean ↓", color="#1a1f2e")
          .highlight_max("Cosine ↑", color="#1a2e1a")
          .highlight_min("Manhattan ↓", color="#1f1a2e")
          .format({"Euclidean ↓": "{:.4f}", "Cosine ↑": "{:.4f}", "Manhattan ↓": "{:.4f}"}),
        width="stretch", hide_index=True,
    )
