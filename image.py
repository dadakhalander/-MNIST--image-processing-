import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys

# Check and install missing packages
try:
    import tensorflow
except ImportError:
    st.error("TensorFlow not installed. Please wait while we install dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    st.rerun()

# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

@st.cache_resource
def load_model(model_type):
    try:
        if model_type == "ReLU":
            return tf.keras.models.load_model('models/best_relu_model.h5')
        return tf.keras.models.load_model('models/best_tanh_model.h5')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def preprocess_image(image):
    try:
        img = np.array(image.convert('L'))
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 10
            digit = img[max(0, y-padding):min(img.shape[0], y+h+padding),
                       max(0, x-padding):min(img.shape[1], x+w+padding)]
        else:
            digit = img
        
        h, w = digit.shape
        scale = 20 / max(h, w)
        resized = cv2.resize(digit, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        
        pad_top = (28 - resized.shape[0]) // 2
        pad_bottom = 28 - resized.shape[0] - pad_top
        pad_left = (28 - resized.shape[1]) // 2
        pad_right = 28 - resized.shape[1] - pad_left
        
        padded = cv2.copyMakeBorder(
            resized, 
            pad_top, pad_bottom, 
            pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=0
        )
        
        return (padded.astype('float32') / 255.0).reshape(1, 784), padded
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None

def main():
    st.title("Handwritten Digit Classifier")
    st.write("Upload an image of a handwritten digit (0-9)")
    
    model_type = st.sidebar.radio("Model Type:", ("ReLU", "Tanh"))
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
            processed_image, processed_vis = preprocess_image(image)
            if processed_image is None:
                return
                
            st.image(processed_vis, caption="Processed Image", width=150)
            
            model = load_model(model_type)
            if model is None:
                return
                
            prediction = model.predict(processed_image)
            pred_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Digit", pred_digit)
                st.metric("Confidence", f"{confidence*100:.2f}%")
            with col2:
                fig, ax = plt.subplots()
                ax.bar(range(10), prediction[0])
                ax.set_xticks(range(10))
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
