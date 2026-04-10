import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, InceptionV3
try:
    from tensorflow.keras.applications import EfficientNetV2B0
except ImportError:
    from tensorflow.keras.applications import EfficientNetB0
    EfficientNetV2B0 = None

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os


# Configuration
DATASET_DIR = r'd:\Project\Digital images of defective and good condition tyres'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

def build_model(model_name):
    input_shape = (224, 224, 3)
    
    if model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'EfficientNetV2B0':
        if EfficientNetV2B0 is not None:
            base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            print("EfficientNetV2B0 not found, falling back to EfficientNetB0")
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50V2':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Freeze base model
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Data Generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    models_to_train = ['MobileNetV2', 'EfficientNetV2B0', 'ResNet50V2', 'InceptionV3']
    history_dict = {}
    best_accuracies = {}

    print(f"Starting comparison of {len(models_to_train)} models...")

    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        model = build_model(model_name)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f'model_{model_name.lower()}.h5', 
                monitor='val_accuracy', 
                save_best_only=True, 
                mode='max',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        history_dict[model_name] = history.history
        best_val_acc = max(history.history['val_accuracy'])
        best_accuracies[model_name] = best_val_acc
        print(f"Finished {model_name}. Best Validation Accuracy: {best_val_acc:.4f}")

    # Plot Comparison
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for model_name, hist in history_dict.items():
        plt.plot(hist['val_accuracy'], label=f'{model_name} (Best: {max(hist["val_accuracy"]):.2f})')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    for model_name, hist in history_dict.items():
        plt.plot(hist['val_loss'], label=model_name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nComparison plot saved as model_comparison.png")

    # Save Summary
    with open('comparison_summary.txt', 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("========================\n")
        for model_name, acc in best_accuracies.items():
            f.write(f"{model_name}: {acc*100:.2f}%\n")
            
    print("Summary saved to comparison_summary.txt")

if __name__ == "__main__":
    main()
