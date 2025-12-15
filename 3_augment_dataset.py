import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

RAW_DIR = Path("dataset_raw_alphabets")
AUG_DIR = Path("dataset_augmented")
AUG_DIR.mkdir(exist_ok=True)

# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=15,         # rotate +/- 15 degrees
    width_shift_range=0.1,     # horizontal shift
    height_shift_range=0.1,    # vertical shift
    shear_range=0.1,           # shearing
    zoom_range=0.1,            # zoom in/out
    brightness_range=[0.7, 1.3], # brightness change
    horizontal_flip=True,      # flip horizontally
    fill_mode='nearest'        # fill in missing pixels after transformations
)

AUG_PER_IMAGE = 5

for label_folder in RAW_DIR.iterdir():
    if label_folder.is_dir():
        label = label_folder.name
        print(f"Augmenting label: {label}")
        save_folder = AUG_DIR / label
        save_folder.mkdir(parents=True, exist_ok=True)

        for img_file in label_folder.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            img = load_img(img_file)  
            x = img_to_array(img)     
            x = x.reshape((1,) + x.shape)  

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=save_folder,
                                      save_prefix=label,
                                      save_format='jpg'):
                i += 1
                if i >= AUG_PER_IMAGE:
                    break

print("Augmentation complete! Augmented dataset saved in 'dataset_augmented/'")
