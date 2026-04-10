import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import csv
import sys
import gc

# Configuration
TEST_DIR = r'd:\Project\TEST'
OUTPUT_CSV = 'comparison_predictions.csv'
IMG_SIZE = (224, 224)

# Models to compare
MODELS = {
    'MobileNetV2': 'model_mobilenetv2.h5',
    'EfficientNetV2B0': 'model_efficientnetv2b0.h5',
    'ResNet50V2_Base': 'model_resnet50v2.h5',
    'ResNet50V2_Finetuned': 'model_resnet50v2_finetuned.h5'
}

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory '{TEST_DIR}' not found.")
        return

    # 1. Get images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_extensions)]
    files.sort()
    
    if not files:
        print("No images found in TEST folder.")
        return
        
    print(f"Found {len(files)} images in {TEST_DIR}.")

    # Initialize results dictionary with filenames
    # Structure: { 'filename.jpg': { 'Filename': 'filename.jpg', 'ModelA_Label': ..., ... } }
    full_results = {f: {'Filename': f} for f in files}

    # 2. Iterate through models sequentially (Load -> Predict -> Unload)
    for model_name, model_path in MODELS.items():
        print(f"\n" + "="*40)
        print(f"Processing Model: {model_name}")
        print(f"Loading from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found. Skipping.")
            continue
            
        try:
            # Load Model
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully. Starting predictions...")
            
            # Predict for all images
            for i, filename in enumerate(files):
                img_path = os.path.join(TEST_DIR, filename)
                
                try:
                    # Preprocess
                    img = image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0
                    
                    # Predict
                    pred_prob = model.predict(img_array, verbose=0)[0][0]
                    
                    is_good = pred_prob > 0.5
                    label = 'Good' if is_good else 'Defective'
                    display_conf = pred_prob if is_good else 1 - pred_prob
                    
                    # Store in results dict
                    full_results[filename][f'{model_name}_Label'] = label
                    full_results[filename][f'{model_name}_Conf'] = f"{display_conf:.4f}"
                    full_results[filename][f'{model_name}_Raw'] = f"{pred_prob:.4f}"
                    
                except Exception as e:
                    print(f"Error predicting {filename} with {model_name}: {e}")
            
            print(f"Finished predictions for {model_name}.")
            
            # Cleanup memory
            del model
            tf.keras.backend.clear_session()
            gc.collect()
            print("Memory cleared.")
            
        except Exception as e:
            print(f"Critical error loading/running {model_name}: {e}")

    # 3. Save to CSV
    print(f"\n" + "="*40)
    print("Saving results...")
    
    if full_results:
        # Determine fieldnames dynamically
        # Start with Filename, then add columns for each processed model
        fieldnames = ['Filename']
        for model_name in MODELS.keys():
            # Check if this model was actually processed (has keys in the first result)
            first_key = list(full_results.keys())[0]
            if f'{model_name}_Label' in full_results[first_key]:
                fieldnames.extend([f'{model_name}_Label', f'{model_name}_Conf', f'{model_name}_Raw'])
            
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(full_results.values())
            print(f"Results saved to '{OUTPUT_CSV}'")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
