import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Configuration (Consistent with train.py)
DATASET_DIR = r'd:\Project\Digital images of defective and good condition tyres'
MODEL_PATH = 'model_resnet50v2_finetuned.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train.py first.")
        return

    # 1. Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Setup Validation Data Generator (Must be identical to train.py)
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False  # Important for confusion matrix
    )

    # 3. Predict on Validation Set
    print("Generating predictions...")
    predictions = model.predict(val_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = val_generator.classes
    class_names = list(val_generator.class_indices.keys())

    # 4. Calculate Metrics
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Tire Quality')
    
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved as {cm_path}")
    
    # 6. Overall Accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()
