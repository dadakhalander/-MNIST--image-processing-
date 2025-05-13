import tensorflow as tf
from tensorflow.keras.models import save_model

def save_models(relu_model, tanh_model):
    """Save trained models to disk"""
    relu_model.save('models/best_relu_model.h5')
    tanh_model.save('models/best_tanh_model.h5')
    print("Models saved successfully!")
