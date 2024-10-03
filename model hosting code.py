import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Function to preprocess the image
def preprocess_image(img):
    # Resize the image to the required input shape of the model
    img = img.resize((256, 256))
    # Convert the image to array
    img_array = np.array(img)
    # Normalize the pixel values to range [0, 1]
    img_array = img_array / 255.0
    # Add batch dimension to the image array
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_colony_count(img, model):
    # Preprocess the image
    img_array = preprocess_image(img)
    # Perform prediction using the model
    prediction = model.predict(img_array)
    return prediction[0][0]

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = r"model path"
    model = tf.keras.models.load_model(model_path)
    return model

# Main function to run the web application
def main():
    st.title("Colony Count Prediction - SV/PK")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Check if predict button is clicked
        if st.button("Predict"):
            # Load the model
            model = load_model()

            # Perform prediction
            prediction = predict_colony_count(image, model)

            # Display the predicted count
            st.success(f"Predicted colony count: {prediction}")

if __name__ == "__main__":
    main()
