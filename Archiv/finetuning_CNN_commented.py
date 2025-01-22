import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from collections import Counter

# Function for applying Mixup data augmentation
# Mixup generates a convex combination of two samples and their labels to improve generalization
# `alpha` controls the strength of the mix
# Returns mixed inputs, target labels for both samples, and the mixing coefficient

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)  # Sampling the mixing coefficient from a beta distribution
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)  # Randomly shuffle batch indices
    mixed_x = lam * x + (1 - lam) * x[index, :]  # Create mixed inputs
    y_a, y_b = y, y[index]  # Labels for both components of the mix
    return mixed_x, y_a, y_b, lam

# Computes the loss for Mixup
# Combines losses for both sets of labels, weighted by the Mixup coefficient

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Function to fine-tune a pre-trained model
# Freezes earlier layers of the model and updates the fully connected layers for the specific task
# Adds new fully connected layers for the specified number of output classes

def finetune(model, num_classes):
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers initially
    for param in model.layer4.parameters():
        param.requires_grad = True  # Unfreeze the last block (layer4) for fine-tuning

    num_ftrs = model.fc.in_features  # Number of input features for the original classifier
    # Replace the fully connected layers with a custom architecture
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Function to train and validate the model
# Supports Mixup for the training phase and uses early stopping based on validation accuracy

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=15, patience=3):
    best_model_wts = model.state_dict()  # Save the best model weights
    best_acc = 0.0  # Best validation accuracy
    early_stop_counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()  # Set model to training or evaluation mode
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # Reset gradients

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)  # Apply Mixup
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # Mixup loss
                    else:
                        outputs = model(inputs)  # Forward pass
                        loss = criterion(outputs, labels)  # Standard loss

                    _, preds = torch.max(outputs, 1)  # Predictions from the model
                    if phase == 'train':
                        loss.backward()  # Backpropagation
                        optimizer.step()  # Update weights

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)  # Count correct predictions

            if phase == 'train':
                scheduler.step()  # Update learning rate

            epoch_loss = running_loss / len(dataloader.dataset)  # Average loss for the epoch
            epoch_acc = running_corrects.double() / len(dataloader.dataset)  # Average accuracy for the epoch
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                early_stop_counter = 0  # Reset early stopping counter
            elif phase == 'val':
                early_stop_counter += 1  # Increment early stopping counter

        # Stop training early if validation performance does not improve
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)  # Load the best model weights
    return model

# Function to evaluate the model on test data
# Computes the overall accuracy

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    accuracy = 0
    with torch.no_grad():  # Disable gradient computation
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)  # Forward pass
            y_hat = F.softmax(y_hat, dim=1).detach().cpu().numpy()  # Convert to probabilities
            y = y.cpu().numpy()
            y_hat = np.argmax(y_hat, axis=1)  # Predicted class
            accuracy += (y_hat == y).mean()  # Accuracy for this batch
    accuracy /= len(test_loader)  # Average accuracy across batches
    return accuracy

# Main script to set up and train the model
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    print(f'Using device: {device}')

    # Data augmentation pipeline for training images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ResNet)
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.RandomRotation(30),  # Random rotation up to 30 degrees
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Random color adjustments
        transforms.RandomAffine(degrees=0, shear=10),  # Random affine transformations
        transforms.RandomGrayscale(p=0.2),  # Convert to grayscale with 20% probability
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    # Load dataset from the specified directory
    dataset = datasets.ImageFolder(root="C:/Users/kej9ho/Documents/Bilderkennung/data/train", transform=data_transforms)

    # Split the dataset into training (80%) and testing (20%) subsets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Create data loaders for training and testing
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Calculate class weights to handle class imbalance
    class_counts = Counter([label for _, label in dataset])  # Count occurrences of each class
    class_weights = [1.0 / class_counts[i] for i in range(len(dataset.classes))]  # Inverse frequency of each class
    class_weights = torch.tensor(class_weights).to(device)

    # Load a pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = finetune(model, len(dataset.classes))  # Fine-tune for the specific task
    model = model.to(device)  # Move model to the selected device

    # Set up the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted cross-entropy loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)  # Adam optimizer
    scheduler = CosineAnnealingLR(optimizer, T_max=15)  # Learning rate scheduler with cosine annealing

    # Train the model
    num_epochs = 15
    model = train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs)

    # Evaluate the model on the test dataset
    accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test data: {accuracy:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
