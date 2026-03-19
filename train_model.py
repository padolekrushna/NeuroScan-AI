#!/usr/bin/env python
# coding: utf-8
"""
Fixed training script for Brain MRI Tumor Classification.
Uses the local brain-tumor-mri-dataset folder instead of a hardcoded kagglehub path.
Saves the trained model to model/model.h5
"""

import os
import sys
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle

# ── Configuration ──────────────────────────────────────────────────────────────

# Use paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(SCRIPT_DIR, 'brain-tumor-mri-dataset', 'Training')
TEST_DIR  = os.path.join(SCRIPT_DIR, 'brain-tumor-mri-dataset', 'Testing')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

# ── Validate dataset paths ──────────────────────────────────────────────────────

for path in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(path):
        print(f"ERROR: Dataset folder not found: {path}")
        sys.exit(1)

print(f"DONE: Training directory: {TRAIN_DIR}")
print(f"DONE: Testing  directory: {TEST_DIR}")

# ── TensorFlow import (after path check) ───────────────────────────────────────

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

print(f"DONE: TensorFlow version: {tf.__version__}")

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_image_paths_and_labels(data_dir, random_state=42):
    paths = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            paths.append(image_path)
            labels.append(label)
    paths, labels = shuffle(paths, labels, random_state=random_state)
    return paths, labels


train_paths, train_labels = load_image_paths_and_labels(TRAIN_DIR)
test_paths, test_labels   = load_image_paths_and_labels(TEST_DIR)

print(f"DONE: Training samples : {len(train_paths)}")
print(f"DONE: Testing  samples : {len(test_paths)}")
print(f"DONE: Classes          : {sorted(set(train_labels))}")

# ── Augmentation & Preprocessing ───────────────────────────────────────────────

def augment_image(image):
    image = image.squeeze()
    image = Image.fromarray(np.uint8(image))
    if random.random() < 0.3:
        image = image.rotate(random.uniform(-7, 7))
    if random.random() < 0.4:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return image


def open_images(paths):
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='grayscale')
        image = np.array(image)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        image = augment_image(image)
        image = np.repeat(image, 3, axis=-1)
        images.append(image.astype('float32'))
    return np.array(images)


def encode_label(labels):
    unique_labels = sorted(os.listdir(TRAIN_DIR))
    encoded = [unique_labels.index(label) for label in labels]
    return np.array(encoded)


def datagen(paths, labels, batch_size=12, epochs=1):
    for _ in range(epochs):
        combined = list(zip(paths, labels))
        random.shuffle(combined)
        paths, labels = zip(*combined)
        for i in range(0, len(paths), batch_size):
            batch_paths  = paths[i:i + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = encode_label(labels[i:i + batch_size])
            yield batch_images, batch_labels

# ── Model Architecture ─────────────────────────────────────────────────────────

print("\nBuilding model...")

base_model = VGG16(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers:
    if 'block5' in layer.name:
        layer.trainable = True

num_classes = len(sorted(os.listdir(TRAIN_DIR)))

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

model.summary()

# ── Training ───────────────────────────────────────────────────────────────────

steps_per_epoch = len(train_paths) // BATCH_SIZE

print(f"\nStarting training: {EPOCHS} epochs, {steps_per_epoch} steps/epoch ...")

history = model.fit(
    datagen(train_paths, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
)

# ── Save Model ─────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"\nDONE: Model saved to: {MODEL_PATH}")

# ── Quick summary ──────────────────────────────────────────────────────────────

final_acc  = history.history['sparse_categorical_accuracy'][-1]
final_loss = history.history['loss'][-1]
print(f"DONE: Final training accuracy : {final_acc:.4f}")
print(f"DONE: Final training loss     : {final_loss:.4f}")
print("\nDONE: Training complete! The API is now ready to serve predictions.")
