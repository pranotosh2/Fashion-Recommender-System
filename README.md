# Fashion-Recommender-System
Deep Learning Image Based Fashion Recommendations 

* (KNN & FAISS – Feature Similarity Search)

* This project implements a simple image-based recommendation system using deep learning feature extraction and similarity search.

* Users upload an image, and the system recommends visually similar images from a dataset.
* Image folder link: [Image Link](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

# 🚀 Features

Deep feature extraction using ResNet50

Two similarity search options:

KNN (Cosine Similarity) – simple & beginner-friendly

FAISS (L2 / Cosine) – fast & scalable

Streamlit web interface

Permanent feature storage (.npy, .pkl)

No training required (uses pretrained CNN)

## 🧠 Workflow
```
Images Dataset  
↓  
ResNet50 Feature Extraction  
↓  
2048-D Feature Vectors  
↓  
embeddings.npy + filenames.pkl  
↓  
KNN / FAISS Similarity Search  
↓  
Recommended Images
```
RUN: python feature_extraction.py
```
This will generate:

* embeddings.npy → shape (N, 2048)

* filenames.pkl → image paths

# KNN Version
```
streamlit run knn.py
```

# FAISS Version
```
streamlit run app.py
```
