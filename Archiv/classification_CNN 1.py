import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

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

class PredictionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = sorted(os.listdir("C:/Users/kej9ho/Documents/Bilderkennung/data/train"))
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = finetune(model, len(class_names))
    model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    pred_data = PredictionDataset(root="C:/Users/kej9ho/Documents/Bilderkennung/data/pred", transform=data_transforms)
    pred_loader = DataLoader(pred_data, batch_size=32, shuffle=False, num_workers=4)

    prediction_dir = r"C:\Users\kej9ho\Documents\Bilderkennung\data\predictions"
    shutil.rmtree(prediction_dir, ignore_errors=True)
    os.makedirs(prediction_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for inputs, paths in pred_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = F.softmax(outputs, dim=1).cpu().numpy()
            pred_labels = np.argmax(preds, axis=1)
            confidences = np.max(preds, axis=1)  # höchste Wahrscheinlichkeit jeder Vorhersage
            
            for i, path in enumerate(paths):
                pred_class = class_names[pred_labels[i]]
                confidence = confidences[i] * 100  # in Prozent umwandeln
                class_dir = os.path.join(prediction_dir, pred_class)
                os.makedirs(class_dir, exist_ok=True)
                
                # Neues Dateiname-Format: Ursprünglicher Name + Wahrscheinlichkeit
                filename = os.path.basename(path)
                new_filename = f"{os.path.splitext(filename)[0]}_{confidence:.2f}%.jpg"
                new_path = os.path.join(class_dir, new_filename)
                
                shutil.copy(path, new_path)

    print('Predictions with confidence scores copied to prediction directory.')
