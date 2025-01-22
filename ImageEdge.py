import cv2
import numpy as np

# Input image path
input_path = '123.jpg'  # Replace with the path to your image
output_path = '1234.jpg'  # Path to save the resulting image

# Load the image
image = cv2.imread(input_path)
if image is None:
    print("Image could not be loaded.")
    exit()

# Resize the image to 512x512
image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using the Canny edge detector
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Thicken the edges
kernel = np.ones((3, 3), np.uint8)
thick_edges = cv2.dilate(edges, kernel, iterations=1)

# Create a white image
result = np.full_like(gray, 255)  # White background

# Mark the edges as black on the white image
result[thick_edges > 0] = 0  # Add black edges

# Save the result
cv2.imwrite(output_path, result)

print(f"Image with black edges on a white background has been saved at: {output_path}")
