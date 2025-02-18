import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras import models

def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title("Mango Classification")

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
model = models.load_model('mango.h5')

if image_file is not None:
    st.image(load_image(image_file), width=250)
    image = Image.open(image_file)
    image = image.resize((100, 100))
    image_arr = np.array(image.convert('RGB'))
    image_arr = image_arr.reshape((1, 100, 100, 3))
    
    result = model.predict(image_arr)
    ind = np.argmax(result)
    
    classes = ['Healthy', 'Rotten']
    st.header(f"Prediction: {classes[ind]}")
    st.header('Confidence: ' + str(round(np.max(result), 4) * 100) + ' %')
