import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
MODEL_PATH = 'model_resnet152V2.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    st.write(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255.0  # Scaling
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        return "The leaf is diseased cotton leaf"
    elif preds == 1:
        return "The leaf is diseased cotton plant"
    elif preds == 2:
        return "The leaf is fresh cotton leaf"
    else:
        return "The leaf is fresh cotton plant"

st.title("Cotton Leaf Disease Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    prediction = model_predict(uploaded_file, model)
    st.write("Prediction:", prediction)
