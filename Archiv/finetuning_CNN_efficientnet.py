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

# Mixup Funktion
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Funktion zum Fine-Tuning des Modells
def finetune(model, num_classes):
    for param in model.parameters():
        param.requires_grad = False
    # Letzte Schicht anpassen
    num_ftrs = model.classifier[1].in_features  # EfficientNet hat eine 'classifier' Schicht
    model.classifier = nn.Sequential(
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

# Trainingsfunktion
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=15, patience=3):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    early_stop_counter = 0

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
                    if phase == 'train':
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
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
                early_stop_counter = 0
            elif phase == 'val':
                early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

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

    # Datenaugmentierung
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root="C:/Users/kej9ho/Documents/Bilderkennung/data/train", transform=data_transforms)

    # Zufälliger Split in Trainings- und Testdaten (80:20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Klassen-Gewichte berechnen
    class_counts = Counter([label for _, label in dataset])
    class_weights = [1.0 / class_counts[i] for i in range(len(dataset.classes))]
    class_weights = torch.tensor(class_weights).to(device)

    # EfficientNet model laden
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    model = finetune(model, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=15)

    num_epochs = 15
    model = train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs)

    accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test data: {accuracy:.4f}')

    torch.save(model.state_dict(), 'model.pth')
