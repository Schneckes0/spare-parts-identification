import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from PIL import Image
import matplotlib.pyplot as plt

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs):
    since = time.time()

    # Bestes Modell speichern
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoche {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Jede Epoche hat eine Trainings- und eine Validierungsphase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Trainingsmodus
            else:
                model.eval()   # Evaluationsmodus

            running_loss = 0.0
            running_corrects = 0

            # Daten iterieren
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Gradienten zurücksetzen
                optimizer.zero_grad()

                # Vorwärtslauf
                # Nur im Trainingsmodus werden die Gradienten berechnet
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Rückwärtslauf und Optimierung nur im Trainingsmodus
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistiken sammeln
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Verlust: {epoch_loss:.4f} Genauigkeit: {epoch_acc:.4f}')

            # Kopieren des Modells, wenn es besser ist
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training abgeschlossen in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Beste Validierungsgenauigkeit: {best_acc:.4f}')

    # Bestes Modell laden
    model.load_state_dict(best_model_wts)
    return model

def predict_image(model, device, class_names, data_transforms, image_path):
    # Bild laden und transformieren
    image = Image.open(image_path)
    image = data_transforms['val'](image).unsqueeze(0)
    image = image.to(device)

    # Vorhersage
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    class_idx = preds.item()
    class_name = class_names[class_idx]
    print(f'Vorhergesagte Klasse: {class_name}')

    # Bild anzeigen
    plt.imshow(Image.open(image_path))
    plt.title(f'Vorhergesagt: {class_name}')
    plt.axis('off')
    plt.show()

def main():
    # 1. Konfiguration

    # Pfad zum Datenverzeichnis
    data_dir = 'data'  # Ändere dies auf den Pfad zu deinem Datenordner

    # Anzahl der Klassen in deinem Datensatz
    num_classes = 3  # Ändere dies entsprechend der Anzahl deiner Klassen

    # Anzahl der Trainingsepochen
    num_epochs = 25  # Ändere dies nach Bedarf

    # Batch-Größe für Training und Validierung
    batch_size = 4

    # Lernrate
    learning_rate = 0.001

    # Gerät konfigurieren (CPU oder GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Verwende Gerät: {device}')

    # 2. Datenvorbereitung

    # Datenvorverarbeitung und Augmentationen
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    }

    # Daten laden
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # Datenlader erstellen
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }

    # Datensatzgrößen
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Klassennamen
    class_names = image_datasets['train'].classes
    print(f'Klassen: {class_names}')

    # 3. Modell laden und anpassen

    # Vortrainiertes ResNet-50-Modell laden
    model = models.resnet50(pretrained=True)

    # Den Fully Connected Layer anpassen
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Modell auf Gerät übertragen
    model = model.to(device)

    # Verlustfunktion und Optimierer definieren
    criterion = nn.CrossEntropyLoss()

    # Alle Parameter werden trainiert
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Lernratenplaner
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Training ausführen

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, device, num_epochs)

    # 5. Modell speichern

    model_path = 'best_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Modell gespeichert unter: {model_path}')

    # 6. Beispielvorhersage

    # Pfad zu einem Testbild
    test_image_path = 'test/key1.jpg'  # Ändere dies auf den Pfad zu deinem Testbild

    # Vorhersage ausführen
    predict_image(model, device, class_names, data_transforms, test_image_path)

if __name__ == '__main__':
    main()
