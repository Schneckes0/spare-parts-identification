import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Gerät konfigurieren (CPU oder GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Verwende Gerät: {device}')

# Anzahl der Klassen
num_classes = 3

# Modellarchitektur definieren und Gewichte laden
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model = model.to(device)
model.eval()
print("Modell geladen und bereit für Vorhersagen.")

# Klassennamen definieren
class_names = ['Energiekette', 'Gleitlager','Drehkranz']  # Passen Sie die Reihenfolge an
print(f"Klassennamen: {class_names}")

# Bildtransformationen definieren
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Bild laden und vorverarbeiten
def load_image(image_path):
    input_image = Image.open(image_path)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch, input_image

# Vorhersagefunktion
def predict_image(model, image_path):
    input_batch, original_image = load_image(image_path)
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx]
    confidence = probabilities[predicted_idx].item()
    print(f"Vorhergesagte Klasse: {predicted_class}, Vertrauen: {confidence*100:.2f}%")
    plt.imshow(original_image)
    plt.title(f"Vorhergesagt: {predicted_class} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

# Hauptfunktion
def main():
    image_path = 'test/screencapture (24).png'  # Ersetzen Sie dies durch den Pfad zu Ihrem Bild
    predict_image(model, image_path)

if __name__ == '__main__':
    main()
