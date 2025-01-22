import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

LABEL_SMOOTHING = 0.1

def finetune(model, num_classes):
    """
    Fine-tune a ResNet50 model.
    Unfreezes additional layers (layer2, layer3, layer4).
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last three blocks (layer2, layer3, layer4) and BatchNorm
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.bn1.parameters():
        param.requires_grad = True

    # Replace the classifier layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model

def train_model(model, criterion, optimizer, scheduler, train_ds, val_ds, num_epochs, use_plateau=False):
    """
    Trains the given model for 'num_epochs' epochs.
    If use_plateau=True, the ReduceLROnPlateau scheduler is used.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Use AMP only on GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluation mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Enable gradient computation only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    if scaler is not None and phase == 'train':
                        # Use AMP if available and in training
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # CPU or no AMP
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Update the scheduler
            if phase == 'train' and not use_plateau:
                scheduler.step()
            elif phase == 'val' and use_plateau:
                scheduler.step(epoch_loss)

            # Save best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # ------------------------------------
    # Data augmentation & transformations
    # ------------------------------------
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Read class names from folder structure
    class_names = sorted(os.listdir("data"))
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print('Class names saved to class_names.json')

    # ---------------------------------------
    # Load dataset & split (80% train / 30% val)
    # ---------------------------------------
    full_ds = datasets.ImageFolder(root="data", transform=train_transforms)
    train_size = int(0.80 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Apply validation transforms to the validation dataset
    val_ds.dataset.transform = val_transforms

    print(f"Dataset sizes -> train: {train_size}, val: {val_size}")

    # -------------------------
    # Load & customize ResNet50
    # -------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = finetune(model, num_classes=len(class_names))
    model = model.to(device)

    # CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Use AdamW optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,
        weight_decay=1e-4
    )

    # Use ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3, 
        verbose=True
    )

    # Number of epochs
    num_epochs = 25

    # Train (train/val only). use_plateau=True to use ReduceLROnPlateau
    model = train_model(
        model, criterion, optimizer, scheduler,
        train_ds, val_ds,
        num_epochs=num_epochs,
        use_plateau=True
    )

    # Save the model weights
    torch.save(model.state_dict(), 'model_resnet50.pt')
    print("Model weights saved successfully: model_resnet50.pt")
