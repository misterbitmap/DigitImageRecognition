import numpy as np
import pandas as pd

from activationHelper import relu,softmax

#The 10,000 test cases from the MNIST dataset
#These datasets are available for download in: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

# Load the MNIST dataset
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

#The first column contains the image label and the remaining columns contains the pixel value.

# Split data into features (X) and labels (y)
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Print some statistics to verify data
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
print(f"Unique labels in y_train: {np.unique(y_train)}")

# Define the architecture
input_size = 784
hidden_size = 128
output_size = 10

# Initialize weights and biases
np.random.seed(42)  # for reproducibility
weights1 = np.random.randn(input_size, hidden_size) * 0.01
weights2 = np.random.randn(hidden_size, output_size) * 0.01
bias1 = np.zeros((1, hidden_size))
bias2 = np.zeros((1, output_size))



# Training parameters
learning_rate = 0.001
epochs = 600
batch_size = 32

for epoch in range(epochs):
    # Shuffle the data
    shuffle_index = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_index]
    y_train = y_train[shuffle_index]
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward Propagation
        hidden_layer = relu(np.dot(X_batch, weights1) + bias1)
        output_layer = softmax(np.dot(hidden_layer, weights2) + bias2)

        # Backward Propagation
        output_error = output_layer - np.eye(output_size)[y_batch]
        hidden_error = np.dot(output_error, weights2.T) * (hidden_layer > 0)

        # Gradient calculation
        grad_weights2 = np.dot(hidden_layer.T, output_error) / batch_size
        grad_bias2 = np.mean(output_error, axis=0)
        grad_weights1 = np.dot(X_batch.T, hidden_error) / batch_size
        grad_bias1 = np.mean(hidden_error, axis=0)

        # Weight updates
        weights2 -= learning_rate * grad_weights2
        bias2 -= learning_rate * grad_bias2
        weights1 -= learning_rate * grad_weights1
        bias1 -= learning_rate * grad_bias1

    # Print accuracy every 10 epochs
    if epoch % 10 == 0:
        hidden_layer = relu(np.dot(X_train, weights1) + bias1)
        output_layer = softmax(np.dot(hidden_layer, weights2) + bias2)
        predictions = np.argmax(output_layer, axis=1)
        accuracy = np.mean(predictions == y_train)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy:.3f}')

# Test the network
hidden_layer_test = relu(np.dot(X_test, weights1) + bias1)
output_layer_test = softmax(np.dot(hidden_layer_test, weights2) + bias2)

predictions_test = np.argmax(output_layer_test, axis=1)
accuracy_test = np.mean(predictions_test == y_test)
print(f'Test Accuracy: {accuracy_test:.3f}')


# Save the trained model
np.save('trained_model/weights1.npy', weights1)
np.save('trained_model/weights2.npy', weights2)
np.save('trained_model/bias1.npy', bias1)
np.save('trained_model/bias2.npy', bias2)