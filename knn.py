import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


os.makedirs("uploads", exist_ok=True)

st.title("Fashion Recommender System")

feature_list = np.load("embeddings.npy")

with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)


base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])


def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()
    features = features / norm(features)
    return features

def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=11,
        algorithm="brute",
        metric="cosine"
    )
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices[0]


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = save_uploaded_file(uploaded_file)

    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

    features = feature_extraction(img_path, model)

    indices = recommend(features, feature_list)

    st.subheader("Recommended Images")

    cols = st.columns(10)
    for i in range(10):
        with cols[i]:
            st.image(filenames[indices[i]])
