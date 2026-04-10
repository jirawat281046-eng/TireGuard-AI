import os
# --- SAFETY MODE: FORCE CPU & DISABLE HANGING OPTS ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Reduce logging noise

import numpy as np
print("Importing TensorFlow... (Please wait)")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = r'd:\Project\Digital images of defective and good condition tyres'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

def main():
    print("\n" + "="*50)
    print("STEP 1: SCANNING DATASET DIRECTORIES...")
    print("="*50)
    
    classes = ['defective', 'good', 'not_tire']
    found_all = True
    for cls in classes:
        path = os.path.join(DATASET_DIR, cls)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"Found class '{cls}': {count} images")
        else:
            print(f"ERROR: Directory for class '{cls}' NOT found at {path}")
            found_all = False
    
    if not found_all:
        print("\nStopping: Folders are missing. Please ensure 'not_tire' exists.")
        return

    print("\n" + "="*50)
    print("STEP 2: PREPARING DATA GENERATORS...")
    print("="*50)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    print("Loading Training data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', seed=42
    )

    print("Loading Validation data...")
    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False, seed=42
    )

    print("\n" + "="*50)
    print("STEP 3: CALCULATING CLASS WEIGHTS...")
    print("="*50)
    
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    
    # Boost 'not_tire' importance
    not_tire_idx = train_generator.class_indices.get('not_tire')
    if not_tire_idx is not None:
        class_weight_dict[not_tire_idx] *= 5.0
        print(f"Boosting 'not_tire' (Index {not_tire_idx}) weight to compensate low count.")

    print("\n" + "="*50)
    print("STEP 4: INITIALIZING NEURAL NETWORK...")
    print("="*50)
    
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model compilation complete.")

    checkpoint = ModelCheckpoint('model_resnet50v2_multiclass.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    print("\n" + "="*50)
    print("STEP 5: TRAINING STARTING NOW!")
    print("="*50)
    
    history = model.fit(
        train_generator, validation_data=val_generator, epochs=EPOCHS,
        class_weight=class_weight_dict, callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print("\nDone! Saving plots...")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy']); plt.title('Acc'); plt.grid()
    plt.subplot(1, 2, 2); plt.plot(history.history['loss']); plt.title('Loss'); plt.grid()
    plt.savefig('multiclass_training_history.png')
    
    print("\n[SUCCESS] Final model saved as 'model_resnet50v2_multiclass.h5'")

if __name__ == "__main__":
    main()
