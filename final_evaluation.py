import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATASET_DIR = r'd:\Project\Digital images of defective and good condition tyres'
MODEL_PATH = r'd:\Project\model_resnet50v2_finetuned.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print("Starting Final Evaluation...")
    
    # 1. Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Setup Validation Generator (Standard Rescale only)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # 3. Predict
    print(f"Evaluating {len(val_gen.filenames)} images...")
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = (y_pred_probs > 0.5).astype('int32').flatten()
    
    # 4. Metrics
    report = classification_report(y_true, y_pred, target_names=['Defective', 'Good'], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. Speed Test
    import time
    start = time.time()
    for _ in range(5): # warm up
        _ = model.predict(val_gen.next()[0], verbose=0)
    
    start = time.time()
    _ = model.predict(val_gen.next()[0], verbose=0)
    inf_time = (time.time() - start) * 1000 / BATCH_SIZE
    
    # 6. Output Results
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (POST-TYRENET)")
    print("="*50)
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall:    {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score:  {report['weighted avg']['f1-score']:.4f}")
    print(f"Inference: {inf_time:.2f} ms/image")
    print("="*50)
    
    # Save Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Defective', 'Good'], yticklabels=['Defective', 'Good'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Final Confusion Matrix (with TyreNet)')
    plt.savefig('final_confusion_matrix.png')
    print("Confusion Matrix saved as final_confusion_matrix.png")

if __name__ == "__main__":
    main()
