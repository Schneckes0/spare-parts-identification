from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import uuid

# Initialize the Flask application
app = Flask(__name__)

# Configuration settings for file uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory for uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # Allowed file extensions

# Configure the device for PyTorch (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# --------------------------------------------------
# Load and customize the ResNet50 model
# --------------------------------------------------

# Load class names from a JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
num_classes = len(class_names)  # Total number of output classes

# Load a pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer for the new number of classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),                 # Add dropout for regularization
    nn.Linear(num_features, 512),   # Reduce the number of features to 512
    nn.ReLU(),                      # Apply ReLU activation
    nn.Dropout(0.4),                 # Another dropout layer
    nn.Linear(512, num_classes)      # Final layer matches the number of classes
)

# Path to the saved model weights
model_path = 'model_resnet50.pt'

# Load model weights
try:
    state_dict = torch.load(model_path, map_location=device)  # Load weights to the correct device
    model.load_state_dict(state_dict)  # Apply the weights to the model
    print("ResNet50 model loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    exit(1)  # Exit if there is a runtime error
except Exception as e:
    print(f"Unexpected error occurred: {e}")
    exit(1)

# Move the model to the appropriate device and set it to evaluation mode
model = model.to(device)
model.eval()
print("Model is ready for predictions.")

# --------------------------------------------------
# Define image transformations
# --------------------------------------------------
# Preprocessing pipeline for images (ResNet50 requires 3-channel RGB input)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),          # Convert image to a PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalization for ResNet50
        std=[0.229, 0.224, 0.225]
    ),
])

# Check if a file is allowed based on its extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --------------------------------------------------
# Route for the home page
# --------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Ensure a file is present in the request
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected.')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Save the file
            # Perform prediction
            result, top5 = predict_image(model, filepath)
            # Display the results
            return render_template('result.html', filename=filename, result=result, top5=top5)
        else:
            return render_template('index.html', error='Invalid file.')
    return render_template('index.html')

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_image(model, image_path):
    input_image = Image.open(image_path).convert('RGB')  # Ensure it's a 3-channel image
    input_tensor = preprocess(input_image)  # Apply transformations
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(input_batch)  # Run inference
        probabilities = torch.softmax(output, dim=1)[0]  # Apply softmax for probabilities
        _, predicted_idx = torch.max(probabilities, dim=0)  # Find the class with the highest probability

        # Get the top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    # Map the predicted index to class name
    predicted_class = class_names[predicted_idx.item()]
    confidence = probabilities[predicted_idx].item()

    top5 = []  # Store top 5 predictions
    for i in range(top5_prob.size(0)):
        class_idx = top5_idx[i].item()
        class_name = class_names[class_idx]
        prob = top5_prob[i].item()
        top5.append({'class': class_name, 'probability': f"{prob*100:.2f}%"})

    # Prepare the result for the top prediction
    result = {'predicted_class': predicted_class, 'confidence': f"{confidence*100:.2f}%"}
    return result, top5

# --------------------------------------------------
# Main Flask application
# --------------------------------------------------
if __name__ == '__main__':
    # Ensure the upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Run the Flask app on host 0.0.0.0 and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
