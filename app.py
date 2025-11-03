import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model('mask_model.h5')

# Streamlit UI
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")
st.title("😷 Face Mask Detection using CNN")
st.write("Upload an image to check if the person is wearing a mask or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and convert to RGB
    img = Image.open(uploaded_file).convert("RGB")   # ✅ ensures 3 channels
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    label = "Without Mask ❌" if prediction[0][0] > 0.5 else "With Mask ✅"

    st.markdown(f"### Prediction: {label}")
