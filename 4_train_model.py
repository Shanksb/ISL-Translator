import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

RAW_DIR = Path("dataset_raw_alphabets")
AUG_DIR = Path("dataset_augmented")

DATA_DIR = Path("dataset_combined")
DATA_DIR.mkdir(exist_ok=True)

import shutil
import os

for source_dir in [RAW_DIR, AUG_DIR]:
    for label_folder in source_dir.iterdir():
        if label_folder.is_dir():
            dest_folder = DATA_DIR / label_folder.name
            dest_folder.mkdir(parents=True, exist_ok=True)
            for img_file in label_folder.iterdir():
                shutil.copy(img_file, dest_folder)

print("Dataset combined. Starting training...")

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% validation
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# CNN Model
num_classes = 26   

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
EPOCHS = 25
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save the trained model
model.save("asl_alphabet_model.h5")
print("Training complete! Model saved as 'asl_alphabet_model.h5'")
