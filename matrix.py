#!/bin/env python3

import keras
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class DataLoaderWrapper(keras.utils.Sequence):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):
        batch = next(iter(self.dataloader))
        inputs, labels = batch
        inputs = inputs.permute(0, 2, 3, 1).numpy()  # Change shape to (batch_size, height, width, channels)
        labels = keras.utils.to_categorical(labels.numpy(), num_classes=len(self.dataset.classes))  # Convert to one-hot encoding
        return inputs, labels

# Set the number of threads
torch.set_num_threads(4)  # Limit to 4 threads

# Define the image input size
input_size = (64, 64)

# Define the image transformation to resize the images
val_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

# Load the datasets
val_dataset = datasets.ImageFolder(root='generations/val', transform=val_transform)
test_dataset = datasets.ImageFolder(root='generations/test', transform=test_transform)

# Create data loaders
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Function to load a pre-trained model
def load_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError("Unknown model name")

    model.load_state_dict(torch.load(f'models/{model_name}.pth'))
    model.eval()
    return model

# Function to compute confusion matrix
def compute_confusion_matrix(model, dataloader, classes):
    all_preds = []
    all_labels = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    return cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.title(title)
    plt.show()

# Assuming you have the classes list
classes = val_dataset.classes

# Load the models and generate confusion matrices
model_names = ['resnet18', 'densenet121', 'mobilenet_v3_small']
num_classes = len(classes)

for model_name in model_names:
    model = load_model(model_name, num_classes)

    # Compute confusion matrix for validation dataset
    val_cm = compute_confusion_matrix(model, val_loader, classes)
    plot_confusion_matrix(val_cm, classes, title=f'{model_name.capitalize()} Validation Confusion Matrix')

    # Compute confusion matrix for test dataset
    test_cm = compute_confusion_matrix(model, test_loader, classes)
    plot_confusion_matrix(test_cm, classes, title=f'{model_name.capitalize()} Test Confusion Matrix')

# Load custom model
model = keras.models.load_model('models/custom.h5')

all_preds = np.argmax(model.predict(DataLoaderWrapper(val_loader)), axis=1)
all_labels = [label for _, label in DataLoaderWrapper(val_loader)]

cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
plot_confusion_matrix(cm, classes, title='Custom Model Validation Confusion Matrix')

all_preds = np.argmax(model.predict(DataLoaderWrapper(test_loader)), axis=1)
all_labels = [label for _, label in DataLoaderWrapper(test_loader)]

cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
plot_confusion_matrix(cm, classes, title='Custom Model Test Confusion Matrix')
