import numpy as np
from activationHelper import relu, softmax
from imgHelper import preprocess_image

# Load the trained model
weights1 = np.load('trained_model/weights1.npy')
weights2 = np.load('trained_model/weights2.npy')
bias1 = np.load('trained_model/bias1.npy')
bias2 = np.load('trained_model/bias2.npy')


image_path = 'images/image9.jpg'

try:
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Using the trained model to make a prediction
    hidden_layer = relu(np.dot(processed_image, weights1) + bias1)
    output_layer = softmax(np.dot(hidden_layer, weights2) + bias2)

    # Get the prediction
    prediction = np.argmax(output_layer)
    print(f"The predicted digit is: {prediction}")

    confidence = np.max(output_layer) * 100
    print(f"Confidence: {confidence:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")