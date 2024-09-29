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
model13 = load_model('Model-A.h5', compile=False, custom_objects={'CustomLayer': CustomLayer})
model14 = load_model('Model-B.h5', compile=False, custom_objects={'CustomLayer': CustomLayer})

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
    st.write("<div style='font-size: 16px; font-family: Arial, sans-serif;'><i>Developed by Nikesh Acharya | MS Data Science Capstone Project | Robert Morris University </i></div>", unsafe_allow_html=True)
    st.markdown("[Click here for GitHub Link](https://github.com/nikAcharya1/streamlit-ML-Nikesh)")
    st.title("Facial Image Classification App")
    st.write("<div style='font-size: 24px;'>This application allows a user to upload a facial image and it uses pre-trained AI models to classify if a person has eyeglasses or not. Additionally, user can provide feedback based on classification results to compare model performances. </div>", unsafe_allow_html=True)
    st.write("")
    st.write("<div style='font-size: 20px;'>Some FYIs:</div>", unsafe_allow_html=True)
    
    # Define your list
    my_list = ["User uploaded images are not saved. So, no privacy concerns.",
        "For better results, upload face image with face covering majority of the image area."]

    # Convert the list items to a bulleted list using HTML tags
    font_size = "20px" 
    bulleted_list = f"<ul style='font-size: {font_size};'>" + "".join([f"<li>{item}</li>" for item in my_list]) + "</ul>"
    
    # Display the bulleted list using markdown
    st.markdown(bulleted_list, unsafe_allow_html=True)
        
    st.write("<hr style='margin: 10px;'>", unsafe_allow_html=True) # Horizontal line
    uploaded_file = st.file_uploader("**Choose a facial image...**", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, width=200, caption="Uploaded Image.")
        st.write("")
        st.write("**Results of Binary Image Classification (1 - Eyeglasses Present, 0 - No Eyeglasses):**")

        # Preprocess and classify
        image_array = preprocess_image(uploaded_file)
        predictions13, predictions14 = predict_image(image_array)

        # Display classification results
        st.write("Model A =", round(predictions13[0][0]))
        st.write("Model B =", round(predictions14[0][0]))

        # Select option and submit entry
        # Define a list of options
        options1 = ["0 - No Eyeglasses", "1 - Eyeglasses Present"]
        options2 = ["Model A", "Model B", "Neither"]

        st.write("")
        st.write("<hr style='margin: 5px;'>", unsafe_allow_html=True) # Horizontal line
        st.write("<div style='text-align: center; font-size: 24px;'>User Feedback on Model Accuracy</div>", unsafe_allow_html=True)
        
        selected_option1 = st.radio("Select correct class:", options1)
        st.write("Select the models that correctly classified:")
        
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
            new_data = pd.DataFrame({'Image ID': uploaded_file.name,
                                    'Actual Class': selected_option1,
                                     'Models that correctly classified': selected_options2})
            import os
            import io
            from github import Github

            # Use Github token without displaying it publicly
            token = st.secrets["github_token"]
            g = Github(token)
            
            # Access repository
            repo = g.get_repo("nikAcharya1/streamlit-ML-Nikesh")

            # Download the spreadsheet file
            contents = repo.get_contents("user_feedback.csv")
            d_data = io.BytesIO(contents.decoded_content)
            data = pd.read_csv(d_data)
            
            # Handle form submission and modify the data
            data = pd.concat([data, new_data], ignore_index=True)
            
            # Convert DataFrame back to CSV format
            csv_data = data.to_csv(index=False)
            
            # Upload modified file to GitHub
            repo.update_file("user_feedback.csv", "Updated by Streamlit form submission", csv_data, contents.sha)
            
            st.write("Data updated.")
            st.write("<hr style='margin: 10px;'>", unsafe_allow_html=True) # Horizontal line
            
          # Load existing data from CSV
            contents = repo.get_contents("user_feedback.csv")
            d_data = io.BytesIO(contents.decoded_content)
            data = pd.read_csv(d_data)
            total_responses = len(data)
            st.write("<div style='text-align: center; font-size: 22px;'><b>Pie Chart Showing Correct Classifications Each Model Made So far:</b></div>", 
                     unsafe_allow_html=True)
            st.write("")
            st.write(f"<div style='text-align: center; font-size: 20px;'>Total User Responses = {total_responses}</div>", 
                     unsafe_allow_html=True)

            # Create a pie chart for 'Models that correctly classified'
            if not data.empty:
                # Count the occurrences of each value in 'Models that correctly classified' column
                model_counts = data['Models that correctly classified'].value_counts()

                # Plot the pie chart
                fig, ax = plt.subplots()

                # Plot pie chart with labels inside
                ax.pie(model_counts, labels=model_counts.index, startangle=90, autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(model_counts) / 100))
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
            else:
                st.write("No data available to create a pie chart.")
               
if __name__ == "__main__":
    main()
