
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
DATASET_DIR = r'd:\Project\Digital images of defective and good condition tyres'
OUTPUT_PATH = r'C:\Users\Jirawat\.gemini\antigravity\brain\2d69d89e-87f9-4667-a620-c342860defe8\tire_augmentation_sample.png'

def generate_dramatic_6_augmentations():
    # 1. Use a clear defective tire image
    img_path = os.path.join(DATASET_DIR, 'defective', 'Defective (1).jpg')
    
    # 2. Load and prep image
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x_batch = x.reshape((1,) + x.shape)
    
    # 3. Define DRAMATIC generators for visualization purposes
    # We use larger ranges than the actual training to make the "concept" visually clear
    gens = {
        'Rotation': ImageDataGenerator(rotation_range=90, fill_mode='nearest'),
        'Width Shift': ImageDataGenerator(width_shift_range=0.5, fill_mode='nearest'),
        'Height Shift': ImageDataGenerator(height_shift_range=0.5, fill_mode='nearest'),
        'Shear': ImageDataGenerator(shear_range=0.6, fill_mode='nearest'),
        'Zoom': ImageDataGenerator(zoom_range=[0.5, 0.5], fill_mode='nearest'),
        'Horizontal Flip': ImageDataGenerator(horizontal_flip=True)
    }
    
    # Generate one sample for each and ensure they aren't the identity transform
    np.random.seed(99) # Change seed for different randomness
    samples = [img]
    
    # Rotation
    samples.append(next(gens['Rotation'].flow(x_batch, batch_size=1))[0].astype('uint8'))
    # Width Shift (Force a shift)
    samples.append(next(gens['Width Shift'].flow(x_batch, batch_size=1))[0].astype('uint8'))
    # Height Shift
    samples.append(next(gens['Height Shift'].flow(x_batch, batch_size=1))[0].astype('uint8'))
    # Shear
    samples.append(next(gens['Shear'].flow(x_batch, batch_size=1))[0].astype('uint8'))
    # Zoom
    samples.append(next(gens['Zoom'].flow(x_batch, batch_size=1))[0].astype('uint8'))
    # Flip (Force a flip by checking)
    aug_flip = next(gens['Horizontal Flip'].flow(x_batch, batch_size=1))[0].astype('uint8')
    samples.append(aug_flip)
    
    # 4. Save individual images for the report
    individual_output_dir = r'C:\Users\Jirawat\.gemini\antigravity\brain\2d69d89e-87f9-4667-a620-c342860defe8\individual_augmentations'
    os.makedirs(individual_output_dir, exist_ok=True)
    
    filenames = [
        '1_original.png',
        '2_rotation.png',
        '3_width_shift.png',
        '4_height_shift.png',
        '5_shear.png',
        '6_zoom.png',
        '7_flip.png'
    ]
    
    for i, sample in enumerate(samples):
        file_path = os.path.join(individual_output_dir, filenames[i])
        plt.imsave(file_path, sample)
        print(f"Saved: {file_path}")

    # 5. Plotting in a 2x4 grid with a cleaner style
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    titles = [
        u'1. ภาพต้นฉบับ\n(Original Image)', 
        u'2. การหมุนภาพ\n(Extreme Rotation)', 
        u'3. การเลื่อนแนวนอน\n(Significant Width Shift)', 
        u'4. การเลื่อนแนวตั้ง\n(Significant Height Shift)',
        u'5. การบิดภาพ\n(High Shear Transformation)',
        u'6. การซูมภาพ\n(Deep Zoom-in)',
        u'7. การพลิกภาพ\n(Horizontal Flip)'
    ]
    
    for i in range(len(samples)):
        axes[i].imshow(samples[i])
        axes[i].set_title(titles[i], fontsize=16, pad=15, fontweight='bold', color='#2c3e50')
        axes[i].axis('off')
        # Add a colored border
        rect = plt.Rectangle((0,0), 223, 223, fill=False, edgecolor='#34495e', linewidth=4)
        axes[i].add_patch(rect)
            
    # Remove the empty last subplot
    axes[7].axis('off')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Dramatic augmentation visualization saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dramatic_6_augmentations()
