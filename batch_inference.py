import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import csv
import sys

# Configuration
MODEL_PATH = 'model_resnet50v2_finetuned.h5'
OUTPUT_CSV = 'prediction_results_finetuned.csv'
IMG_SIZE = (224, 224)

def batch_predict(folder_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    results = []
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(files)} images in {folder_path}. Starting predictions...")
    
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale
            
            # Predict
            prediction = model.predict(img_array, verbose=0)
            confidence = prediction[0][0]
            
            # Determine class
            # 0 -> defective, 1 -> good (Assuming similar to training setup)
            # Threshold at 0.5
            is_good = confidence > 0.5
            label = 'Good' if is_good else 'Defective'
            
            # Format confidence to pure probability of the predicted class
            display_confidence = confidence if is_good else 1 - confidence
            
            print(f"Processed {filename}: {label} ({display_confidence:.2%})")
            
            results.append({
                'Filename': filename,
                'Prediction': label,
                'Confidence': display_confidence,
                'Raw_Score': confidence
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            
    # Save results
    if results:
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Filename', 'Prediction', 'Confidence', 'Raw_Score'])
                writer.writeheader()
                writer.writerows(results)
                
            print("\n" + "="*30)
            print(f"Done! Results saved to '{OUTPUT_CSV}'")
            print(f"Processed: {len(results)} images")
            print("="*30)
        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        # Default to checking if 'TEST' folder exists in current dir
        folder = 'TEST' if os.path.exists('TEST') else None
        
    if folder:
        batch_predict(folder)
    else:
        print("Usage: python batch_inference.py <folder_path>")
