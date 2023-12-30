import streamlit as st
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import pandas as pd

# Load pre-trained model
model11 = load_model('Model1-Run1.h5')
model12 = load_model('Model1-Run2.h5')
model13 = load_model('Model1-Run3.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    load_image = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(load_image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array

def predict_image(image_array):
    predictions11 = model11.predict(image_array)
    predictions12 = model12.predict(image_array)
    predictions13 = model13.predict(image_array)
    return predictions11, predictions12, predictions13

def main():
    st.title("Image Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.")
        st.write("")
        st.write("Classifying...")

        # Preprocess and predict
        image_array = preprocess_image(uploaded_file)
        predictions11, predictions12, predictions13 = predict_image(image_array)

        # Display predictions
        st.write("Model 1 =", round(predictions11[0][0], 2))
        st.write("Model 2 =", round(predictions12[0][0], 2))
        st.write("Model 3 =", round(predictions13[0][0], 2))

        # Select option and submit entry
        
        # Load existing data from CSV (if it exists)
        file_path = 'user_options.csv'
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            data = pd.DataFrame(columns=['Actual Class', 'Most Accurate Model'])

        # Define a list of options
        options1 = ["0", "1"]
        options2 = ["Model 1", "Model 2", "Model 3", "Similar accuracy across all models"]

        # Create a selectbox for multiple choices
        selected_option1 = st.selectbox("Select correct class:", options1)
        selected_option2 = st.selectbox("Select most accurate model:", options2)
        
        # Add a submit button
        if st.button("Submit"):
            # Append the selected option to the DataFrame
            data = data.append({'Actual Class': selected_option1, 
                               'Most Accurate Model': selected_option2}, ignore_index=True)

            # Save the updated DataFrame to the CSV file
            data.to_csv(file_path, index=False)

            st.write("Data updated. You selected:")
            st.write("Actual Class:", selected_option1)
            st.write("Most Accurate Model:", selected_option2)

        # Display the existing data
        # st.write("Existing Data:")
        st.write(data)

if __name__ == "__main__":
    main()
