import streamlit as st
import pickle
import numpy as np
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_features
import torch
import torch.nn as nn
import torchvision.models as models
import tempfile
from keras.models import load_model

st.set_page_config(page_title="Image Classification & Retrieval")
st.title("Similar Image Retrieval")
st.markdown("""
    Welcome to the CIFAR-10 Image Classification and Retrieval system. 
    Upload an image to get its predicted class and find similar images from our dataset.
""")

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

HF_BASE_URL = "https://huggingface.co/datasets/varaiitj/prmldemotest/resolve/main/"

@st.cache_data
def download_and_load_numpy_pickle(filename):
    url = HF_BASE_URL + filename
    response = requests.get(url)
    response.raise_for_status()
    return np.array(pickle.loads(response.content))

@st.cache_data
def download_and_load_pickle(filename):
    url = HF_BASE_URL + filename
    response = requests.get(url)
    response.raise_for_status()
    return pickle.loads(response.content)

@st.cache_resource
def download_and_load_resnet(filename):
    url = HF_BASE_URL + filename
    response = requests.get(url)
    response.raise_for_status()
    with open("temp_resnet.pth", "wb") as f:
        f.write(response.content)
    resnet = models.resnet50(weights=None)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.load_state_dict(torch.load("temp_resnet.pth", map_location=torch.device("cpu")))
    resnet.eval()
    return resnet

@st.cache_resource
def download_and_load_keras_model(filename):
    url = HF_BASE_URL + filename
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_file:
        temp_file.write(response.content)
        temp_file.flush()
        model = load_model(temp_file.name)
    return model

train_features = download_and_load_numpy_pickle("Resnet_train.pkl")
train_images = download_and_load_pickle("RawPixels_train.pkl")
y_train = download_and_load_pickle("Labels_train.pkl")
resnet = download_and_load_resnet("resnet50_feature_extractor.pth")
model = download_and_load_keras_model("model.keras")

uploaded_file = st.file_uploader("Upload a Query Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", use_container_width=True)

    image_np = np.array(query_image)
    query_feature = np.array(extract_features(image_np, resnet)).reshape(1, -1)

    pred_class_idx = int(np.argmax(model.predict(query_feature), axis=1)[0])
    pred_class_label = class_labels[pred_class_idx]
    st.subheader(f"Predicted Class: {pred_class_label}")

    class_indices = [i for i, label in enumerate(y_train) if label == pred_class_idx]
    if not class_indices:
        st.error("No images found for this class in the training set.")
    else:
        filtered_features = train_features[class_indices]
        filtered_images = [train_images[i] for i in class_indices]

        similarities = cosine_similarity(query_feature, filtered_features)[0]
        top5_indices = similarities.argsort()[-5:][::-1]

        top_images = [filtered_images[idx] for idx in top5_indices]

        st.subheader("Top 5 Similar Images:")
        cols = st.columns(len(top_images))
        for i, img in enumerate(top_images):
            with cols[i]:
                st.image(img, width=120)
