import streamlit as st
import numpy as np
import pickle
import faiss
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tempfile

# ---------------- Load stored data ----------------
features = np.load("embeddings.npy")
filenames = pickle.load(open("filenames.pkl", "rb"))

# ---------------- FAISS index ----------------
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)

# ---------------- Model ----------------
base_model = ResNet50(weights="imagenet", include_top=False)
model = Model(base_model.input, GlobalMaxPooling2D()(base_model.output))

# ---------------- Helper ----------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img, verbose=0).flatten().astype("float32")

# ---------------- UI ----------------
st.set_page_config(page_title="Image Recommendation", layout="wide")
st.title("Image-Based Recommendation System")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        query = extract_features(tmp.name)

    D, I = index.search(np.array([query]), 6)

    st.subheader("Uploaded Image")
    st.image(uploaded, width=250)

    st.subheader("Recommended Images")
    cols = st.columns(5)

    for i, idx in enumerate(I[0][1:]):
        with cols[i]:
            st.image(Image.open(filenames[idx]), use_container_width=True)
