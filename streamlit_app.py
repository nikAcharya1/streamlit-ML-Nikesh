# Load necessary libraries 
import streamlit as st
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load pre-trained model
model13 = load_model('Model3.h5')
model14 = load_model('Model4.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    load_image = load_img(image_path, target_size=(224, 224))
    img = img_to_array(load_image)
    img_array = preprocess_input(img.reshape(1,224,224,3))
    return img_array

def predict_image(image_array):
    predictions13 = model13.predict(image_array)
    predictions14 = model14.predict(image_array)
    return predictions13, predictions14

def main():
    st.title("Facial Image Classification App")

    uploaded_file = st.file_uploader("**Choose a facial image...**", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, width=200, caption="Uploaded Image.")
        st.write("")
        st.write("**Results of Binary Image Classification (1 - Eyeglasses Present, 0 - None):**")

        # Preprocess and classify
        image_array = preprocess_image(uploaded_file)
        predictions13, predictions14 = predict_image(image_array)

        # Display classification results
        st.write("Model A =", round(predictions13[0][0]))
        st.write("Model B =", round(predictions14[0][0]))

        # Select option and submit entry
        
        # Load existing data from CSV (if it exists)
        file_path = 'user_options.csv'
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            data = pd.DataFrame(columns=['Actual Class', 'Models that correctly classified'])

        # Define a list of options
        options1 = ["0 - No Eyeglasses", "1 - Eyeglasses Present"]
        options2 = ["Model A", "Model B"]

        selected_option1 = st.selectbox("Select correct class:", options1)
        st.write("Select the model/s that correctly classified:")
        
        # Create a list to store the state of each checkbox
        checkbox_states = {}

        # Iterate over each option and create a checkbox for it
        for option in options2:
            # Use the checkbox function to create a clickable checkbox
            checkbox_states[option] = st.checkbox(option)

        # Add a submit button
        if st.button("Submit"):
            # Filter the selected options based on the checkbox states
            selected_options2 = [option for option, state in checkbox_states.items() if state]

           # Create a new DataFrame with the selected options
            new_data = pd.DataFrame({'Actual Class': selected_option1,
                                     'Models that correctly classified': selected_options2})

            # Concatenate the new data with the existing DataFrame
            data = pd.concat([data, new_data], ignore_index=True)

            # Save the updated DataFrame to the CSV file
            data.to_csv(file_path, index=False)
            
            st.write("Data updated.")
            
            # Load existing data from CSV
            file_path = 'user_options.csv'
            data = pd.read_csv(file_path)
            
            st.title("Breakdown of How Many Correct Classifications Each Model Made So Far:")

            # Create a pie chart for 'Models that correctly classified'
            if not data.empty:
                # Count the occurrences of each value in 'Models that correctly classified' column
                model_counts = data['Models that correctly classified'].value_counts()

                # Plot the pie chart
                fig, ax = plt.subplots()
                ax.pie(model_counts, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
            else:
                st.write("No data available to create a pie chart.")
            
     
if __name__ == "__main__":
    main()
