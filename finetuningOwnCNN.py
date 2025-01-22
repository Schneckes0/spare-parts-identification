import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR
import json
from torch.optim import AdamW

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Erste Convolutional Layer
            nn.Conv2d(1, 32, kernel_size=5, padding=1),  # Output: (32, 220, 220)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (32, 110, 110)
            
            # Zweite Convolutional Layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (64, 110, 110)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (64, 55, 55)
            
            # Dritte Convolutional Layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: (128, 55, 55)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (128, 27, 27)
            
            # Vierte Convolutional Layer (neu hinzugefügt)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: (256, 27, 27)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: (256, 13, 13)
        )
        
        # Berechnung der Ausgabegröße nach CNN
        # 256 Kanäle mit einer Größe von 13x13
        self.flattened_size = 256 * 13 * 13  # 256 * 169 = 43264

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 30)  # 30 Klassen
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.fc(X)
        return X

# Trainingsfunktion mit Validierung
def train_model(model, criterion, optimizer, scheduler, train_ds, val_ds, num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Evaluation auf Testdaten
def evaluate_model(model, test_loader):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat = F.softmax(y_hat, dim=1).detach().cpu().numpy()
            y = y.cpu().numpy()
            y_hat = np.argmax(y_hat, axis=1)
            accuracy += (y_hat == y).mean()
    accuracy /= len(test_loader)
    return accuracy

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Datenaugmentation und Verarbeitung
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Konvertiere Bilder in Graustufen
        transforms.Resize((224, 224)),  # Anpassung auf 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalisierung für Graustufen
    ])


    val_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Konvertiere Bilder in Graustufen
    transforms.Resize((224, 224)),  # Anpassung auf 64x64
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalisierung für Graustufen
    ])

    # Klassen speichern
    class_names = sorted(os.listdir("data/train"))
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    print('Klassennamen gespeichert in class_names.json')

    # Daten laden
    full_ds = datasets.ImageFolder(root="data/train", transform=train_transforms)
    train_size = int(0.7 * len(full_ds))
    val_size = int(0.2 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size])

    # Validation/Test-Datensatz mit anderem Transform
    val_ds.dataset.transform = val_test_transforms
    test_ds.dataset.transform = val_test_transforms

    # Eigenes CNN-Modell initialisieren
    model = CNN()
    model = model.to(device)

    # Hyperparameter und Optimierung
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # Lernrate erhöht
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 

    # Training
    num_epochs = 25
    model = train_model(model, criterion, optimizer, scheduler, train_ds, val_ds, num_epochs)

    # Evaluation
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test data: {accuracy:.4f}')

    # Modell speichern
    torch.save(model.state_dict(), 'model_cnn.pt')
