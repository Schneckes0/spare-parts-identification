import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score

# Funktion zum Fine-Tuning des Modells
def finetune(model, num_classes):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# Trainingsfunktion mit Validierung
def train_model(model, criterion, optimizer, scheduler, train_ds, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        train_size = int(0.8 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_subset, val_subset = random_split(train_ds, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)

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

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    class_names = sorted(os.listdir("data/train"))

    train_ds = datasets.ImageFolder(root="data/train", transform=data_transforms)
    test_ds = datasets.ImageFolder(root="data/val", transform=data_transforms)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = finetune(model, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 25
    model = train_model(model, criterion, optimizer, scheduler, train_ds, num_epochs)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test data: {accuracy:.4f}')

    torch.save(model.state_dict(), 'model.pth')
