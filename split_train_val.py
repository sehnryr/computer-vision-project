#!/bin/env python3

import os
import random
import shutil

# Set the seed for reproducibility
random.seed(42)

# Define paths
dataset_dir = 'generations'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

# Create train and val directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Define classes
classes = ['gen-1', 'gen-2', 'gen-3', 'gen-4', 'gen-5']

# Split ratio (e.g., 80% training, 20% validation)
split_ratio = 0.8

# Split the dataset
for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    images = os.listdir(cls_dir)

    # Shuffle images
    random.shuffle(images)

    # Calculate split index
    split_index = int(len(images) * split_ratio)

    # Split images
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class directories in train and val
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Move images to train directory
    for img in train_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))

    # Move images to val directory
    for img in val_images:
        shutil.move(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))
