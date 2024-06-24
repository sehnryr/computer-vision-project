#!/bin/env python3

import keras
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
import json
import os

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

# Set up logger
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

logger.setLevel(logging.DEBUG)

# Define the image input size
input_size = (64, 64)

# Define the image transformation to resize the images
train_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='generations/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='generations/val', transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the pre-trained models
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
mobilenet_v3_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes in the dataset
num_classes = len(train_dataset.classes)
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
mobilenet_v3_small.classifier[3] = nn.Linear(mobilenet_v3_small.classifier[3].in_features, num_classes)
densenet121.classifier = nn.Linear(densenet121.classifier.in_features, num_classes)

# Move the models to the appropriate device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)
mobilenet_v3_small = mobilenet_v3_small.to(device)
densenet121 = densenet121.to(device)

# Group the models in a dictionary
models = {
    'resnet18': resnet18,
    'mobilenet_v3_small': mobilenet_v3_small,
    'densenet121': densenet121,
}

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the number of epochs
num_epochs = 10

def run(model, data_loader, device, optimizer, criterion, phase='train'):
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    # Iterate over data
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc.item()

def train(model, data_loader, device, optimizer, criterion):
    model.train()
    return run(model, data_loader, device, optimizer, criterion, 'train')

def validate(model, data_loader, device, optimizer, criterion):
    model.eval()
    return run(model, data_loader, device, optimizer, criterion, 'val')

# Initialize lists to store metrics
train_losses = {model_name: [] for model_name in models.keys()}
val_losses = {model_name: [] for model_name in models.keys()}
train_accuracies = {model_name: [] for model_name in models.keys()}
val_accuracies = {model_name: [] for model_name in models.keys()}

# Create custom model using Keras.Sequential
custom_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(input_size[0], input_size[1], 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the custom model
logger.info('Training the custom model')
history = custom_model.fit(
    DataLoaderWrapper(train_loader),
    epochs=num_epochs,
    validation_data=DataLoaderWrapper(val_loader)
)

train_losses['custom'] = history.history['loss']
val_losses['custom'] = history.history['val_loss']
train_accuracies['custom'] = history.history['accuracy']
val_accuracies['custom'] = history.history['val_accuracy']
logger.debug(f'Custom Model Training Losses: {json.dumps(train_losses["custom"])}')
logger.debug(f'Custom Model Validation Losses: {json.dumps(val_losses["custom"])}')
logger.debug(f'Custom Model Training Accuracies: {json.dumps(train_accuracies["custom"])}')
logger.debug(f'Custom Model Validation Accuracies: {json.dumps(val_accuracies["custom"])}')

for model_name, model in models.items():
    logger.info(f'Training {model_name}')
    # Define the optimizer for the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')

        logger.debug(f'Running training phase for epoch {epoch+1}')
        train_loss, train_acc = train(model, train_loader, device, optimizer, criterion)
        logger.debug(f'Running validation phase for epoch {epoch+1}')
        val_loss, val_acc = validate(model, val_loader, device, optimizer, criterion)

        train_losses[model_name].append(train_loss)
        val_losses[model_name].append(val_loss)

        train_accuracies[model_name].append(train_acc)
        val_accuracies[model_name].append(val_acc)

logger.info('Training completed')
logger.debug(f'Training Losses: {json.dumps(train_losses)}')
logger.debug(f'Validation Losses: {json.dumps(val_losses)}')
logger.debug(f'Training Accuracies: {json.dumps(train_accuracies)}')
logger.debug(f'Validation Accuracies: {json.dumps(val_accuracies)}')

# Save the models
logger.debug('Saving the models')
os.makedirs('models', exist_ok=True)
for model_name, model in models.items():
    torch.save(model.state_dict(), f'models/{model_name}.pth')
custom_model.save('models/custom.h5')

# Plot the validation losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for model_name in models.keys():
    plt.plot(val_losses[model_name], label=f'{model_name} Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Losses')

# Plot the validation accuracies
plt.subplot(1, 2, 2)
for model_name in models.keys():
    plt.plot(val_accuracies[model_name], label=f'{model_name} Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.title('Validation Accuracies')

plt.show()
