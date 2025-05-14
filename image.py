import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf

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
        image = image.convert('L')  # Grayscale
        image = ImageOps.invert(image)  # Invert for black background
        image = ImageOps.autocontrast(image)

        # Resize while keeping aspect ratio, then pad to 28x28
        image.thumbnail((20, 20), Image.ANTIALIAS)
        padded = Image.new('L', (28, 28), color=0)
        upper_left = ((28 - image.width) // 2, (28 - image.height) // 2)
        padded.paste(image, upper_left)

        # Normalize and reshape
        img_array = np.array(padded).astype('float32') / 255.0
        return img_array.reshape(1, 784), padded
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None

def main():
    st.title("Handwritten Digit Classifier")
    st.write("Upload an image of a handwritten digit (0–9)")

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
