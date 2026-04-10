import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Configuration
DATASET_DIR = r'd:\Project\Digital images of defective and good condition tyres'
MODEL_PATH = 'model_resnet50v2.h5'
OUTPUT_MODEL_PATH = 'model_resnet50v2_finetuned.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINETUNE_EPOCHS = 15
LEARNING_RATE = 1e-5  # Very low learning rate for fine-tuning

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    # 1. Load the pre-trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Check current summary
    # model.summary()

    # 2. Inspect and Unfreeze
    print(f"Model layers: {[layer.name for layer in model.layers]}")
    
    # Check if we have a nested model (Functional layer)
    base_model_layer = None
    for layer in model.layers:
        if 'resnet' in layer.name or 'mobilenet' in layer.name or 'efficientnet' in layer.name:
            base_model_layer = layer
            break
            
    if base_model_layer and hasattr(base_model_layer, 'layers'):
        print(f"Found nested base model: {base_model_layer.name}")
        base_model_layer.trainable = True
        
        # Unfreeze last 30 layers of the inner model
        layers_to_unfreeze = 30
        for layer in base_model_layer.layers[:-layers_to_unfreeze]:
            layer.trainable = False
        print(f"Unfroze last {layers_to_unfreeze} layers of nested base model.")
        
    else:
        print("Model appears to be flattened (no nested base model found). Processing all layers...")
        # If flattened, we skip the last few layers (Head) and fine-tune the end of the feature extractor.
        # Head layers: Dense(1), Dropout, Dense(128), GlobalAvg
        # We want to unfreeze the layers BEFORE the head.
        
        # Count total layers
        total_layers = len(model.layers)
        print(f"Total layers in model: {total_layers}")
        
        # Heuristic: The head is usually the last 4-5 layers.
        # We want to unfreeze, say, the last 30 layers of the network, but typically we want to keep BatchNormalization frozen for stability.
        # A simple strategy: Unfreeze the last N layers.
        
        layers_to_unfreeze = 40 # Includes head + some top conv blocks
        
        for i, layer in enumerate(model.layers):
            if i < total_layers - layers_to_unfreeze:
                layer.trainable = False
            else:
                # Keep BatchNorm layers frozen for best practice in fine-tuning
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
                    
        print(f"Unfroze last {layers_to_unfreeze} layers (excluding BatchNorm).")

    # 3. Re-compile the model
    # Important: compile after changing trainable status
    print(f"Re-compiling with learning rate {LEARNING_RATE}...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 4. Data Generators (With Advanced Augmentation)
    # Using a common seed to ensure the validation split remains consistent
    common_seed = 42

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,          # Increased rotation (better for tire symmetry)
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.4,             # Increased zoom range
        horizontal_flip=True,
        vertical_flip=True,         # Added vertical flip (tires are often symmetric)
        brightness_range=[0.5, 1.5], # Widened brightness variation
        channel_shift_range=20.0,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Validation generator should NOT have heavy augmentation, only rescale
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        seed=common_seed
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=common_seed
    )

    # Calculate class weights to handle imbalance (Methodology Fix #2)
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Applied Class Weights: {class_weight_dict}")

    # 5. Train (Fine-tune)
    print("Starting fine-tuning...")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL_PATH, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_generator,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    # 6. Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Fine-tuning Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Fine-tuning Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('finetuning_history.png')
    print("Plot saved as finetuning_history.png")
    
    print(f"Fine-tuning complete. Best model saved as {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
