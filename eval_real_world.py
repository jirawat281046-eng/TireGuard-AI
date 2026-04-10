import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import time

# Configuration
MODEL_PATH = 'model_resnet50v2_finetuned.h5'
TEST_DIR = r'd:\Project\TEST'
IMG_SIZE = (224, 224)

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found.")
        return

    print(f"Loading model: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Testing on {len(image_files)} real-world images...\n")

    print("-" * 75)
    print(f"{'Filename':<25} | {'Prediction':<15} | {'Confidence':<12} | {'Time (ms)':<10}")
    print("-" * 75)

    results = []
    
    for img_name in image_files:
        img_path = os.path.join(TEST_DIR, img_name)
        
        start_time = time.time()
        img_array = load_and_preprocess(img_path)
        prediction = model.predict(img_array, verbose=0)[0][0]
        end_time = time.time()
        
        # 0 -> defective, 1 -> good
        is_good = prediction > 0.5
        label = "👍 GOOD" if is_good else "⚠️ DEFECTIVE"
        confidence = prediction if is_good else 1.0 - prediction
        elapsed_ms = (end_time - start_time) * 1000
        
        print(f"{img_name:<25} | {label:<15} | {confidence:8.2%} | {elapsed_ms:8.1f}")
        results.append(is_good)

    print("-" * 75)
    good_count = sum(results)
    defect_count = len(results) - good_count
    print(f"Summary: {good_count} Good, {defect_count} Defective")

if __name__ == "__main__":
    main()
