import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

# Load models (cache to avoid reloading)
@st.cache_resource
def load_model(model_type):
    if model_type == "ReLU":
        return tf.keras.models.load_model('models/best_relu_model.h5')
    else:
        return tf.keras.models.load_model('models/best_tanh_model.h5')

# Preprocess image function
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    img = np.array(img)
    
    # Invert colors (MNIST has white digits on black background)
    img = 255 - img
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize and reshape
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 784)
    
    return img

# Main app
def main():
    st.title("Handwritten Digit Classifier")
    st.write("Upload an image of a handwritten digit (0-9) to classify it using different neural networks.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio("Choose activation function:", ("ReLU", "Tanh"))
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        
        # Load selected model
        model = load_model(model_type)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Digit", value=predicted_digit)
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
        
        with col2:
            # Plot prediction probabilities
            fig, ax = plt.subplots()
            ax.bar(range(10), prediction[0], color='skyblue')
            ax.set_title("Prediction Probabilities")
            ax.set_xlabel("Digit")
            ax.set_ylabel("Probability")
            ax.set_xticks(range(10))
            ax.set_ylim([0, 1])
            st.pyplot(fig)
        
        # Model information
        st.subheader("Model Information")
        st.write(f"Using model with {model_type} activation functions")
        
        # Show model architecture
        st.code(f"""Model Architecture:
Input Layer: 784 neurons (flattened 28x28 image)
Hidden Layer 1: 256 neurons with {model_type} activation
Hidden Layer 2: 128 neurons with {model_type} activation
Hidden Layer 3: 64 neurons with {model_type} activation
Output Layer: 10 neurons with softmax activation""")

if __name__ == "__main__":
    main()
