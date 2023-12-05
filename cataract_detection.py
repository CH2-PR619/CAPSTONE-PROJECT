import cv2
import tkinter as tk
from tkinter import filedialog
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers.experimental import RMSprop
from keras.preprocessing import image

def load_classification_model():
    model_path = 'cataract_detection.h5'
    model = load_model(model_path)
    return model

def eyes_detection(uploaded_image):
    image = cv2.imread(uploaded_image)

    if image is None:
        print(f"Failed to load image: {uploaded_image}")
        return

    # Konversi gambar ke skala keabuan (grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Inisialisasi detektor mata Haar Cascade
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Deteksi mata dalam gambar
    eyes = eye_cascade.detectMultiScale(gray_image)

    if len(eyes) > 0:
        # Gambar kotak di sekitar mata yang terdeteksi
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Tampilkan gambar yang telah diolah
        print('Mata Terdeteksi')
        cataract_detection()
    else:
        print("Tidak ada mata yang terdeteksi.")
        main()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cataract_detection(model, image):
    image = np.array(image.resize((94, 55)))
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)
    if result > 0.5:
        return 'normal'
    else:
        return 'cataract'

def main():
    st.title("Cataract Detection")

    # Unggah gambar dari pengguna
    uploaded_image = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        eyes_detection(uploaded_image)

        # Lakukan klasifikasi
        model = load_classification_model()
        result = cataract_detection(model, image)

        # Tampilkan hasil klasifikasi
        st.write("Hasil Klasifikasi:", result)

if __name__ == "__main__":
    main()