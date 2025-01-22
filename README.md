
# Spare Parts Identification Using AI

This project aims to develop a system for the automatic identification of spare parts through image data. 
By leveraging artificial intelligence and machine learning, the system seeks to recognize and classify spare parts in images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Contributors](#contributors)
- [License](#license)

## Overview

In various industries, the swift and accurate identification of spare parts is crucial for the maintenance and repair of machinery and equipment. 
This project utilizes advanced image processing techniques and deep neural networks to identify and classify spare parts in images.

## Features

- **Image Preprocessing**: Application of edge detection and other techniques to enhance image quality.
- **Model Training**: Fine-tuning of pre-trained Convolutional Neural Networks (e.g., ResNet50) on specific spare part images.
- **Text Recognition**: Extraction of textual information from images to aid identification.
- **3D Rendering**: Use of Blender to create 3D representations of spare parts.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Schneckes0/spare-parts-identification.git
   ```

2. **Install Dependencies**:

   Navigate to the project directory and install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Blender**:

   Ensure that Blender is installed on your system and that the path to `blender.exe` is set in the environment variables.

## Usage

- **Image Preprocessing**:

  Use the `ImageEdge.py` script to detect edges in input images.

  ```bash
  python ImageEdge.py --input <input_image_path> --output <output_image_path>
  ```

- **Model Training**:

  Utilize `FinetuningCNNResnet50.py` to train the ResNet50 model with your own data.

  ```bash
  python FinetuningCNNResnet50.py --data <dataset_path> --epochs <number_of_epochs>
  ```

- **Text Recognition**:

  Run `ImageRecognitionTextDescription.py` to extract textual information from images.

  ```bash
  python ImageRecognitionTextDescription.py --input <input_image_path>
  ```

- **3D Rendering**:

  Use `RenderGltfWithBlender.py` to render 3D models with Blender.

  ```bash
  python RenderGltfWithBlender.py --model <3d_model_path>
  ```

## Data

The quality and diversity of the training data are critical for the model's performance. 
It is recommended to use an extensive dataset containing images of the spare parts to be identified.

## Model Training

Training an accurate model requires careful data preparation and hyperparameter tuning. 
Use the provided scripts to fine-tune pre-trained models on your specific data.

## Contributors

- [Schneckes0](https://github.com/Schneckes0)

Contributions are welcome! Please open an issue or a pull request to suggest improvements.

## License

This project is licensed under the [MIT License](LICENSE).
