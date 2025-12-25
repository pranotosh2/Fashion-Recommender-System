# Fashion-Recommender-System
Deep Learning Image Based Fashion Recommendations 

🖼️ Image-Based Recommendation System

(KNN & FAISS – Feature Similarity Search)

This project implements a simple image-based recommendation system using deep learning feature extraction and similarity search.

Users upload an image, and the system recommends visually similar images from a dataset.

🚀 Features

Deep feature extraction using ResNet50

Two similarity search options:

KNN (Cosine Similarity) – simple & beginner-friendly

FAISS (L2 / Cosine) – fast & scalable

Streamlit web interface

Permanent feature storage (.npy, .pkl)

No training required (uses pretrained CNN)

Images Dataset
     ↓
ResNet50 (Feature Extraction)
     ↓
Feature Vectors (2048-d)
     ↓
Saved as embeddings.npy
     ↓
Similarity Search (KNN / FAISS)
     ↓
Recommended Images


python feature_extract.py
This will create:

embeddings.npy → shape (N, 2048)

filenames.pkl → image paths

# KNN Version
streamlit run knn.py

# FAISS Version
streamlit run app.py
