import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np

# Load pre-trained model
model = load_model('Model1-Run1.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    load_image = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(load_image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array

def predict_image(image_array):
    predictions = model.predict(image_array)
    return predictions

def main():
    st.title("Image Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess and predict
        image_array = preprocess_image(uploaded_file)
        predictions = predict_image(image_array)

        # Display predictions
        st.write(round(predictions[0][0], 2))

if __name__ == "__main__":
    main()
