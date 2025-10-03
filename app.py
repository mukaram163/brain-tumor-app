import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# ----------------------
# Load Model
# ----------------------
@st.cache_resource
def load_brain_model():
    model = load_model("brain_tumor_cnn_best.h5")
    return model

model = load_brain_model()

# ----------------------
# Class Labels & Info
# ----------------------
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

tumor_info = {
    "Glioma": "A type of tumor that occurs in the brain and spinal cord.",
    "Meningioma": "A tumor that arises from the meninges (the membranes covering the brain & spinal cord).",
    "No Tumor": "No abnormal tumor tissue detected in this MRI image.",
    "Pituitary": "A tumor located in the pituitary gland, affecting hormone balance."
}

# ----------------------
# Helper function
# ----------------------
def predict_mri(img):
    img = img.resize((128, 128))   # Resize to match training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale like training generator

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    return preds[0], class_labels[pred_class]

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide")

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI image to classify into **Glioma, Meningioma, Pituitary, or No Tumor.**")

uploaded_files = st.file_uploader(
    "📥 Upload MRI image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

        # Prediction
        img = image.load_img(uploaded_file, target_size=(128,128))
        preds, pred_label = predict_mri(img)

        st.subheader(f"✅ Prediction: **{pred_label}**")
        st.write(tumor_info[pred_label])

        # Probability chart
        fig, ax = plt.subplots()
        ax.bar(class_labels, preds, color="skyblue")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

# ----------------------
# Sidebar: Model Info
# ----------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write("This model was trained on the **Brain Tumor MRI Dataset**.")
st.sidebar.write("Model: **Custom CNN (Conv2D + Dense layers)**")
st.sidebar.write("Loss: *Categorical Crossentropy* | Optimizer: *Adam*")
st.sidebar.write("Metrics: Accuracy")

st.sidebar.markdown("👨‍💻 Created by [Mukaram Ali] - Connect on [LinkedIn](https://www.linkedin.com/in/mukaram-ali-a05061279/)")
