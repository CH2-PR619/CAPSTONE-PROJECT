import cv2
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image

def load_classification_model():
    model_path = 'cataract_detection.h5'
    model = load_model(model_path)
    return model

def eyes_detection(uploaded_image):
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    if image is None:
        print(f"Failed to load image: {uploaded_image}")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the Haar Cascade eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray_image)

    if len(eyes) > 0:
        # Draw rectangles around the detected eyes
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the processed image
        st.image(image, channels="BGR", caption="Detected Eyes")

        # Perform cataract detection
        model = load_classification_model()
        result = cataract_detection(model, gray_image)
        st.write("Cataract Detection Result:", result)
    else:
        st.write("No eyes detected.")

def cataract_detection(model, gray_image):
    resized_image = cv2.resize(gray_image, (94, 55))
    # Add a channel dimension to the image
    resized_image = np.expand_dims(resized_image, axis=-1)
    # Add a batch dimension (assuming you're processing a single image)
    resized_image = np.expand_dims(resized_image, axis=0)
    # Expand the last axis to have a value of 3
    resized_image = np.repeat(resized_image, 3, axis=-1)

    result = model.predict(resized_image)
    if result > 0.5:
        return 'normal'
    else:
        return 'cataract'

def main():
    st.title("Cataract Detection")

    # Upload an image from the user
    uploaded_image = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        eyes_detection(uploaded_image)

if __name__ == "__main__":
    main()
