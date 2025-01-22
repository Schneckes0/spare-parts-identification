import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from CycleGAN import Generator

# Lade den besten Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_A = Generator().to(device)
G_A.load_state_dict(torch.load('G_A_best.pth', map_location=device))

# Bild vorbereiten (Beispiel: Ein CAD-Bild als Input)
input_image = Image.open('C:/Users/kej9ho/Documents/Bilderkennung/data/CycleGAN/CAD/417.000939/417.000939 (61).png').convert('RGB')

# Transformation anwenden (Resize und zu Tensor konvertieren)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisierung für den Generator
])

input_tensor = transform(input_image).unsqueeze(0).to(device)  # Füge Batch-Dimension hinzu

# Generiere das Bild (CAD -> Real)
with torch.no_grad():  # Wir brauchen keine Gradienten für die Generierung
    generated_image = G_A(input_tensor)

# Bild nach der Generierung zurückverwandeln
generated_image = generated_image.squeeze().cpu().numpy()
generated_image = (generated_image + 1) / 2  # Zurückskalieren (Normalisierung rückgängig machen)

# Bild anzeigen
plt.imshow(generated_image.transpose(1, 2, 0))  # Channels umsortieren (C, H, W -> H, W, C)
plt.axis('off')
plt.show()
