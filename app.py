import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("🧬 Vitamin Deficiency Detection")
st.write("Upload an image to detect vitamin deficiency")

# Load model
model = None
try:
    model = load_model("skin_model.h5")  # ensure model.h5 is in repo
    st.success("Model loaded successfully!")
except:
    st.warning("No model found. Upload model.h5 in repo or via uploader.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img_array = np.expand_dims(np.array(image.resize((224, 224)))/255.0, axis=0)
    prediction = model.predict(img_array)

    class_names = ["Vitamin A","Vitamin B","Vitamin B12","Vitamin C","Vitamin D","Vitamin Zinc","Vitamin D3","Other"]
    predicted_class = class_names[np.argmax(prediction[0])]
    st.success(f"Prediction: {predicted_class}")

