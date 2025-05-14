import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Page configuration
st.set_page_config(page_title="MNIST Digit Classifier (MLP + CNN)", layout="wide")

# Model information
MODEL_INFO = {
    "ReLU": {
        "description": "Multi-Layer Perceptron with ReLU activation",
        "architecture": "784 input → 256 hidden (ReLU) → 128 hidden (ReLU) → 10 output (Softmax)"
    },
    "Tanh": {
        "description": "Multi-Layer Perceptron with Hyperbolic Tangent (tanh) activation",
        "architecture": "784 input → 256 hidden (tanh) → 128 hidden (tanh) → 10 output (Softmax)",
        "note": "tanh (Hyperbolic Tangent) maps inputs to [-1,1], helping model complex patterns"
    },
    "CNN": {
        "description": "Convolutional Neural Network",
        "architecture": "Conv2D(32,3x3) → MaxPool → Conv2D(64,3x3) → MaxPool → Dense(128) → Dense(10, Softmax)"
    }
}

# Load the selected model
@st.cache_resource
def load_model(model_type):
    model_path = f'models/best_{model_type.lower()}_model.h5'
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Preprocess for CNN
def preprocess_for_cnn(image):
    try:
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image)

        image.thumbnail((20, 20), Image.Resampling.LANCZOS)
        padded = Image.new('L', (28, 28), color=0)
        upper_left = ((28 - image.width) // 2, ((28 - image.height) // 2))
        padded.paste(image, upper_left)

        img_array = np.array(padded).astype('float32') / 255.0
        return img_array.reshape(1, 28, 28, 1), padded
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None

# Preprocess for MLP (ReLU or Tanh)
def preprocess_for_mlp(image):
    try:
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image)

        image.thumbnail((20, 20), Image.Resampling.LANCZOS)
        padded = Image.new('L', (28, 28), color=0)
        upper_left = ((28 - image.width) // 2, ((28 - image.height) // 2))
        padded.paste(image, upper_left)

        img_array = np.array(padded).astype('float32') / 255.0
        return img_array.reshape(1, 784), padded
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None

# Main app function
def main():
    st.title("MNIST Digit Classifier")
    st.write("Upload a handwritten digit image (0–9) and select a model to classify it.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Model Configuration")
        model_type = st.radio("Choose Model:", ("ReLU", "Tanh", "CNN"))
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 70)
        
        # Display model information
        st.subheader("Model Information")
        st.write(f"**{model_type} Model**")
        st.write(MODEL_INFO[model_type]["description"])
        st.write(f"**Architecture**: {MODEL_INFO[model_type]['architecture']}")
        if "note" in MODEL_INFO[model_type]:
            st.write(f"**Note**: {MODEL_INFO[model_type]['note']}")

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image, caption="Original Image", width=150)

            # Preprocess image
            if model_type == "CNN":
                processed_image, processed_vis = preprocess_for_cnn(image)
            else:
                processed_image, processed_vis = preprocess_for_mlp(image)

            if processed_image is None:
                return

            # Display processed image
            with col2:
                st.image(processed_vis, caption="Processed Image (28x28)", width=150)

            # Load model
            model = load_model(model_type)
            if model is None:
                return

            # Make prediction with progress bar
            with st.spinner("Classifying image..."):
                prediction = model.predict(processed_image)
                pred_digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100

            # Display results
            st.subheader("Prediction Result")
            col4, col5 = st.columns(2)
            with col4:
                if confidence >= confidence_threshold:
                    st.metric("Predicted Digit", str(pred_digit))
                    st.metric("Confidence", f"{confidence:.2f}%")
                else:
                    st.warning(f"Prediction confidence ({confidence:.2f}%) below threshold ({confidence_threshold}%)")
            
            with col5:
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(range(10), prediction[0], color=['#1f77b4' if i != pred_digit else '#ff7f0e' for i in range(10)])
                ax.set_xticks(range(10))
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Probabilities")
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
                           ha='center', va='bottom')
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
