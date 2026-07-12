# EigenMatch - PCA Based Face Similarity Search

EigenMatch is an image similarity application built using **Principal Component Analysis (PCA)** and the **Eigenfaces** approach. The application finds faces that are visually similar by comparing their feature vectors in PCA space using three different similarity metrics.

**Live Demo:** https://pca-based-image-similarity.streamlit.app/

## Features

- Face similarity search using PCA (Eigenfaces)
- Upload your own face image or select one from the Olivetti Faces dataset
- Compare similarity using:
  - Euclidean Distance
  - Cosine Similarity
  - Manhattan Distance
- Displays Top-N similar faces
- Export search results as CSV
- Interactive Streamlit interface

---

## Dataset

- **Olivetti Faces Dataset**
- 400 grayscale face images
- 40 individuals
- 10 images per person
- Image size: **64 × 64 pixels**

---

## Project Workflow

1. Load and preprocess the face images.
2. Apply **Principal Component Analysis (PCA)** to reduce image dimensions while preserving important facial features.
3. Project every image into the PCA feature space (Eigenfaces).
4. Compare the query image with all gallery images using three similarity metrics.
5. Display the most similar faces.

---

# Similarity Metrics

## 1. Euclidean Distance

Measures the straight-line distance between two feature vectors.

**Formula**

\[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

- Lower distance = More similar images

---

## 2. Cosine Similarity

Measures the angle between two vectors instead of their distance.

**Formula**

\[
\text{Cosine Similarity}=
\frac{x \cdot y}{||x||\,||y||}
\]

- Values range from **-1 to 1**
- Higher value = More similar images

---

## 3. Manhattan Distance

Measures the sum of absolute differences between two vectors.

**Formula**

\[
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
\]

- Lower distance = More similar images

---

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Matplotlib

---

## Run the Project

Install the dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## Project Structure

```
PCA-Based-Image-Similarity
│
├── app/
│   └── app.py
│
├── src/
│   └── face_similarity.py
│
├── notebooks/
│   └── main.ipynb
│
├── models/
│
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Support larger image datasets
- Add deep learning-based face embeddings (FaceNet/ArcFace)
- Improve retrieval accuracy using advanced feature extraction
- Deploy the application on Streamlit Cloud
