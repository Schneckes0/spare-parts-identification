import os
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

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

# Custom Dataset for loading images for prediction
# Each image is loaded and preprocessed based on the provided transformations
class PredictionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Collect all image paths in the specified directory
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)  # Return the total number of images

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # Get the image path at the specified index
        image = Image.open(img_path).convert("RGB")  # Open the image and convert it to RGB format
        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        return image, img_path  # Return the image tensor and its path

if __name__ == '__main__':
    # Set the device for computation (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load class names from the training data directory
    class_names = sorted(os.listdir("C:/Users/kej9ho/Documents/Bilderkennung/data/train"))

    # Load a pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = finetune(model, len(class_names))  # Fine-tune for the specific task

    # Load trained model weights
    model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)  # Move model to the selected device

    # Define data transformations for prediction images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ResNet)
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    # Load prediction data using the custom dataset
    pred_data = PredictionDataset(root="C:/Users/kej9ho/Documents/Bilderkennung/data/pred", transform=data_transforms)
    pred_loader = DataLoader(pred_data, batch_size=32, shuffle=False, num_workers=4)

    # Define directory to store predictions
    prediction_dir = r"C:\Users\kej9ho\Documents\Bilderkennung\data\predictions"
    shutil.rmtree(prediction_dir, ignore_errors=True)  # Clear existing predictions
    os.makedirs(prediction_dir, exist_ok=True)  # Create prediction directory

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():  # Disable gradient computation
        for inputs, paths in pred_loader:
            inputs = inputs.to(device)  # Move inputs to the selected device
            outputs = model(inputs)  # Forward pass
            preds = F.softmax(outputs, dim=1).cpu().numpy()  # Convert to probabilities
            pred_labels = np.argmax(preds, axis=1)  # Predicted class indices
            confidences = np.max(preds, axis=1)  # Highest confidence for each prediction

            for i, path in enumerate(paths):
                pred_class = class_names[pred_labels[i]]  # Get the class name for the prediction
                confidence = confidences[i] * 100  # Convert confidence to percentage
                class_dir = os.path.join(prediction_dir, pred_class)  # Create a subdirectory for each class
                os.makedirs(class_dir, exist_ok=True)  # Ensure the class directory exists

                # Format new filename with the original name and confidence score
                filename = os.path.basename(path)
                new_filename = f"{os.path.splitext(filename)[0]}_{confidence:.2f}%.jpg"
                new_path = os.path.join(class_dir, new_filename)

                shutil.copy(path, new_path)  # Copy the file to the new location with the updated name

    print('Predictions with confidence scores copied to prediction directory.')
