import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
MODEL_PATH = 'model_resnet50v2_multiclass.h5'
IMG_DIR = 'ภาพยางเพิ่มเติม'
OUTPUT_CSV = 'real_world_results.csv'
IMG_SIZE = (224, 224)
CLASSES = ['defective', 'good', 'not_tire']

def load_and_predict():
    print(f"--- Loading Model: {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    
    results = []
    
    print(f"--- Processing Images in {IMG_DIR} ---")
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not img_files:
        print(f"No images found in {IMG_DIR}")
        return

    for img_name in img_files:
        img_path = os.path.join(IMG_DIR, img_name)
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale to [0,1]
        
        # Predict
        preds = model.predict(img_array, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]
        
        results.append({
            'filename': img_name,
            'prediction': CLASSES[class_idx],
            'confidence': confidence,
            'defective_prob': preds[0][0],
            'good_prob': preds[0][1],
            'not_tire_prob': preds[0][2]
        })
        print(f"Processed {img_name}: {CLASSES[class_idx]} ({confidence:.2f})")

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Summary Stats
    summary = df['prediction'].value_counts()
    print("\n--- Prediction Summary ---")
    print(summary)
    
    # Plotting Summary
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='prediction', palette='viridis')
    plt.title('Distribution of Real-world Predictions')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.savefig('real_world_summary_plot.png')
    print("Summary plot saved as 'real_world_summary_plot.png'")

if __name__ == "__main__":
    load_and_predict()
