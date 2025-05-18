import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf

# Page configuration
st.set_page_config(page_title="MNIST Digit Classifier (MLP + CNN + ResNet)", layout="wide")

# Load the selected model
@st.cache_resource
def load_model(model_type):
    try:
        if model_type == "ReLU":
            return tf.keras.models.load_model('models/mnist_relu_model.h5')
        elif model_type == "Tanh":
            return tf.keras.models.load_model('models/best_tanh_model.h5')
        elif model_type == "CNN":
            return tf.keras.models.load_model('models/best_cnn_model.h5')
        elif model_type == "ResNet":
            return tf.keras.models.load_model('models/resnet_mnist_model.h5')
        else:
            st.error("Unknown model type.")
            return None
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
        upper_left = ((28 - image.width) // 2, (28 - image.height) // 2)
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
        upper_left = ((28 - image.width) // 2, (28 - image.height) // 2)
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

    model_type = st.sidebar.radio("Choose Model:", ("ReLU", "Tanh", "CNN", "ResNet"))
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=150)

            if model_type == "CNN" or model_type == "ResNet":
                processed_image, processed_vis = preprocess_for_cnn(image)
            else:
                processed_image, processed_vis = preprocess_for_mlp(image)

            if processed_image is None:
                return

            st.image(processed_vis, caption="Processed Image (28x28)", width=150)

            model = load_model(model_type)
            if model is None:
                return

            prediction = model.predict(processed_image)
            pred_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Digit", str(pred_digit))
                st.metric("Confidence", f"{confidence*100:.2f}%")
            with col2:
                fig, ax = plt.subplots()
                ax.bar(range(10), prediction[0])
                ax.set_xticks(range(10))
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
