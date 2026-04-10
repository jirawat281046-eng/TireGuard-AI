import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

def predict_tire_quality(img_path, model_path='model_resnet50v2_finetuned.h5'):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        return

    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale like in training

    # Predict
    prediction = model.predict(img_array)
    
    # Map prediction to class name
    # Based on train.py, defective might be 0 and good might be 1 (alphabetical order)
    # We will assume: 0 -> defective, 1 -> good
    class_names = ['defective', 'good']
    
    confidence = prediction[0][0]
    result_class = 1 if confidence > 0.5 else 0
    
    print("-" * 30)
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Quality: {class_names[result_class].upper()}")
    print(f"Confidence: {confidence if result_class == 1 else 1-confidence:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
    else:
        predict_tire_quality(sys.argv[1])
