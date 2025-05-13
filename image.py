import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

# Load models (cache to avoid reloading)
@st.cache_resource
def load_model(model_type):
    if model_type == "ReLU":
        return tf.keras.models.load_model('models/best_relu_model.h5')
    else:
        return tf.keras.models.load_model('models/best_tanh_model.h5')

# Enhanced preprocessing for real-world images
def preprocess_image(image):
    # Convert to grayscale
    img = np.array(image.convert('L'))
    
    # Adaptive thresholding to handle different lighting conditions
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours to extract the digit
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (presumably the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the digit region with padding
        padding = 10
        digit = img[max(0, y-padding):min(img.shape[0], y+h+padding),
                   max(0, x-padding):min(img.shape[1], x+w+padding)]
    else:
        digit = img
    
    # Resize to 28x28 while preserving aspect ratio
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))
    
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to make it 28x28
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized, 
        pad_top, pad_bottom, 
        pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=0
    )
    
    # Normalize and reshape
    processed = padded.astype('float32') / 255.0
    processed = processed.reshape(1, 784)
    
    return processed, padded  # Return both processed and visualization image

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
        
        try:
            # Preprocess and predict
            processed_image, processed_vis = preprocess_image(image)
            
            # Show processed image
            st.subheader("Processed Image")
            fig, ax = plt.subplots()
            ax.imshow(processed_vis, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
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
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try with a clearer image of a single digit.")

if __name__ == "__main__":
    main()
