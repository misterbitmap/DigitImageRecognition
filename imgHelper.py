import numpy as np
import cv2

def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load the image.")
    
    # Resize to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    
    # Invert the image if it has a white background and black digit
    if np.mean(img) > 128:
        img = 255 - img
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Reshape to match the input shape of our model
    img = img.reshape(1, 784)
    
    return img