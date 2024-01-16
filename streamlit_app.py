# Load necessary libraries 
import streamlit as st
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
# Load pre-trained model
model11 = load_model('Model1.h5')
model12 = load_model('Model2.h5')
model13 = load_model('Model3.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    load_image = load_img(image_path, target_size=(224, 224))
    img = img_to_array(load_image)
    img_array = preprocess_input(img.reshape(1,224,224,3))
    return img_array

def predict_image(image_array):
    predictions11 = model11.predict(image_array)
    predictions12 = model12.predict(image_array)
    predictions13 = model13.predict(image_array)
    return predictions11, predictions12, predictions13

def main():
    st.title("Facial Image Classification App")

    uploaded_file = st.file_uploader("**Choose a facial image...**", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.")
        st.write("")
        st.write("**Results of Binary Image Classification (1- Glasses, 0 - None):**")

        # Preprocess and classify
        image_array = preprocess_image(uploaded_file)
        predictions11, predictions12, predictions13 = predict_image(image_array)

        # Display classification results
        st.write("Model 1 =", round(predictions11[0][0]))
        st.write("Model 2 =", round(predictions12[0][0]))
        st.write("Model 3 =", round(predictions13[0][0]))

if __name__ == "__main__":
    main()
