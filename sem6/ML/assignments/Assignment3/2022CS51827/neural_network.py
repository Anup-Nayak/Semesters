"""
COL774 Assignment 3: Decision Trees & Neural Networks

File: decision_tree.py
Author: Anup Lal Nayak
Entry Number: <your entry number>

Description:
This file contains all code for Part I (Decision Trees) of Assignment 3.
It includes loading, preprocessing, model training, evaluation, and analysis
as per parts (a) to (f).

Usage:
Run this file as a script after placing the dataset in the appropriate folders.
"""

# ==== Imports ====
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.neural_network import MLPClassifier


# Global variables for data
X_train = None
y_train = None

X_test = None   

output_folder_path = None

# ==== Utility Functions ====
def load_image(path, size=(28, 28)):
    img = Image.open(path)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    return img_array.flatten()  # shape: (2352,)

def load_data(train_path, test_path):
    """Load and preprocess the data."""
    train_folder = train_path
    X_train = []
    y_train = []

    # Loop through each class folder
    for class_id in sorted(os.listdir(train_folder)):
        class_path = os.path.join(train_folder, class_id)
        if not os.path.isdir(class_path):
            continue
        label = int(class_id)
        for fname in os.listdir(class_path):
            img_path = os.path.join(class_path, fname)
            try:
                img_vector = load_image(img_path)
                X_train.append(img_vector)
                y_train.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    
    X = []
    file_names = sorted(os.listdir(test_path))
    for file_name in file_names:
        file_path = os.path.join(test_path, file_name)
        try:
            img = Image.open(file_path).convert('RGB')
            img_resized = img.resize((28, 28))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_flat = img_array.flatten()
            X.append(img_flat)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    X = np.array(X)
    X_test = pd.DataFrame(X)
    
    
    return X_train, y_train, X_test

# Neural Network

class NeuralNetwork:
    def __init__(self, n_features, hidden_layers, n_classes, learning_rate=0.01, batch_size=32,activation='sigmoid'):
        """
        Initialize the neural network
        
        Parameters:
        n_features (int): Number of input features
        hidden_layers (list): List of integers representing number of neurons in each hidden layer
        n_classes (int): Number of output classes
        learning_rate (float): Learning rate for weight updates
        batch_size (int): Mini-batch size for SGD
        """
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation_name = activation
        
        # Initialize network architecture
        self.architecture = [n_features] + hidden_layers + [n_classes]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Xavier/Glorot initialization for weights
        for i in range(len(self.architecture) - 1):
            # Initialize weights with Xavier/Glorot initialization
            w = np.random.randn(self.architecture[i], self.architecture[i+1]) * np.sqrt(2 / (self.architecture[i] + self.architecture[i+1]))
            b = np.zeros((1, self.architecture[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self,x):
        return np.maximum(0, x)

    def relu_derivative(self,x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        
        Parameters:
        X (numpy.ndarray): Input data of shape (batch_size, n_features)
        
        Returns:
        activations (list): List of activations at each layer
        net_inputs (list): List of net inputs at each layer
        """
        activations = [X]  # List to store activations of each layer
        net_inputs = []    # List to store net inputs to each layer
        
        # Hidden layers (sigmoid activation)
        for i in range(len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            net_inputs.append(z)
            if self.activation_name == 'relu':
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        
        # Output layer (softmax activation)
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        net_inputs.append(z_out)
        output = self.softmax(z_out)
        activations.append(output)
        
        return activations, net_inputs
    
    def cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross entropy loss
        
        Parameters:
        y_true (numpy.ndarray): One-hot encoded true labels
        y_pred (numpy.ndarray): Predicted probabilities from softmax
        
        Returns:
        float: Average cross entropy loss
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross entropy loss
        log_likelihood = -np.sum(y_true * np.log(y_pred))
        loss = log_likelihood / y_true.shape[0]
        return loss
    
    def backward(self, X, y, activations, net_inputs):
        """
        Backward propagation to compute gradients
        
        Parameters:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): One-hot encoded true labels
        activations (list): Activations from forward pass
        net_inputs (list): Net inputs from forward pass
        
        Returns:
        tuple: (weight_gradients, bias_gradients)
        """
        m = X.shape[0]  # Batch size
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient calculation
        # delta_L = (output_activations - true_labels)
        delta = activations[-1] - y
        
        # Backpropagate through the network
        for layer in range(len(self.architecture) - 2, -1, -1):
            # Calculate gradients for current layer
            weight_gradients[layer] = np.dot(activations[layer].T, delta) / m
            bias_gradients[layer] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Backpropagate error to previous layer (if not the input layer)
            if layer > 0:
                dA_prev = np.dot(delta, self.weights[layer].T)
                if self.activation_name == 'relu':
                    dZ = dA_prev * self.relu_derivative(net_inputs[layer - 1])
                else:
                    dZ = dA_prev * self.sigmoid_derivative(activations[layer])
                delta = dZ
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update weights and biases using gradients
        
        Parameters:
        weight_gradients (list): Gradients for weights
        bias_gradients (list): Gradients for biases
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train_batch(self, X_batch, y_batch):
        """
        Train on a single mini-batch
        
        Parameters:
        X_batch (numpy.ndarray): Input batch data
        y_batch (numpy.ndarray): One-hot encoded batch labels
        
        Returns:
        float: Loss for this batch
        """
        # Forward pass
        activations, net_inputs = self.forward(X_batch)
        
        # Calculate loss
        loss = self.cross_entropy_loss(y_batch, activations[-1])
        
        # Backward pass
        weight_grads, bias_grads = self.backward(X_batch, y_batch, activations, net_inputs)
        
        # Update parameters
        self.update_parameters(weight_grads, bias_grads)
        
        return loss
    
    def train(self, X_train, y_train, epochs=100, verbose=True,adaptive_lr=False):
        """
        Train the neural network using mini-batch SGD
        
        Parameters:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): One-hot encoded training labels
        epochs (int): Number of training epochs
        verbose (bool): Whether to print progress
        
        Returns:
        list: Training history (losses)
        """
        m = X_train.shape[0]
        train_losses = []
        tol = 1e-6
        prev_loss = None
        
        for epoch in range(epochs):
            # Shuffle training data
            if adaptive_lr:
                current_lr = self.learning_rate / np.sqrt(epoch + 1)
            else:
                current_lr = self.learning_rate
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            num_batches = int(np.ceil(m / self.batch_size))
            epoch_loss = 0
            
            for batch in range(num_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                activations, net_inputs = self.forward(X_batch)
                loss = self.cross_entropy_loss(y_batch, activations[-1])
                weight_grads, bias_grads = self.backward(X_batch, y_batch, activations, net_inputs)

                # Use current (adaptive) learning rate for updates
                for i in range(len(self.weights)):
                    self.weights[i] -= current_lr * weight_grads[i]
                    self.biases[i] -= current_lr * bias_grads[i]
                
                epoch_loss += loss
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            
            if verbose and (epoch % 10 == 0):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
                
            if prev_loss is not None and abs(prev_loss - avg_epoch_loss) < tol:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (Δloss < {tol})")
                break

        prev_loss = avg_epoch_loss
                
        return {"train_losses": train_losses}
    
    def predict(self, X):
        """
        Predict class probabilities for input data
        
        Parameters:
        X (numpy.ndarray): Input data
        
        Returns:
        numpy.ndarray: Predicted class probabilities
        """
        activations, _ = self.forward(X)
        return activations[-1]
    
    def predict_classes(self, X):
        """
        Predict class labels for input data
        
        Parameters:
        X (numpy.ndarray): Input data
        
        Returns:
        numpy.ndarray: Predicted class labels
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on data
        
        Parameters:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): True labels (one-hot encoded)
        
        Returns:
        float: Accuracy
        """
        predictions = self.predict_classes(X)
        true_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        accuracy = np.mean(predictions == true_labels)
        return accuracy

# Utility function to convert labels to one-hot encoding
def to_one_hot(y, num_classes):
    """
    Convert integer labels to one-hot encoded vectors
    
    Parameters:
    y (numpy.ndarray): Array of integer labels
    num_classes (int): Number of classes
    
    Returns:
    numpy.ndarray: One-hot encoded labels
    """
    return np.eye(num_classes)[y]

def part_a():
    pass

def part_b():
    
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    
    y_train_one_hot = to_one_hot(y_train, n_classes)

    
    nn = NeuralNetwork(
        n_features=n_features,
        hidden_layers=[100],  # Single hidden layer
        n_classes=n_classes,
        learning_rate=0.01,
        batch_size=32
    )
    
    nn.train(
        X_train=X_train,
        y_train=y_train_one_hot,
        epochs=100,
        verbose=True
    )
    
    y_test_pred = nn.predict_classes(X_test)
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_b.csv")
    pd.DataFrame({'prediction':y_test_pred}).to_csv(output_path,index=False) 
    
def part_c():
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    
    y_train_one_hot = to_one_hot(y_train, n_classes)

    
    nn = NeuralNetwork(
        n_features=n_features,
        hidden_layers=[512, 256, 128, 64], # Single hidden layer
        n_classes=n_classes,
        learning_rate=0.01,
        batch_size=32
    )
    
    nn.train(
        X_train=X_train,
        y_train=y_train_one_hot,
        epochs=100,
        verbose=True
    )
    
    y_test_pred = nn.predict_classes(X_test)
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_c.csv")
    pd.DataFrame({'prediction':y_test_pred}).to_csv(output_path,index=False) 

def part_d():
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    
    y_train_one_hot = to_one_hot(y_train, n_classes)

    
    nn = NeuralNetwork(
        n_features=n_features,
        hidden_layers=[512, 256, 128, 64], # Single hidden layer
        n_classes=n_classes,
        learning_rate=0.01,
        batch_size=32
    )
    
    nn.train(
        X_train=X_train,
        y_train=y_train_one_hot,
        epochs=100,
        verbose=True,
        adaptive_lr=True
    )
    
    y_test_pred = nn.predict_classes(X_test)
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_d.csv")
    pd.DataFrame({'prediction':y_test_pred}).to_csv(output_path,index=False) 
    
def part_e():
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    y_train_one_hot = to_one_hot(y_train, n_classes)

    
    nn = NeuralNetwork(
        n_features=n_features,
        hidden_layers=[512, 256, 128, 64], # Single hidden layer
        n_classes=n_classes,
        learning_rate=0.01,
        batch_size=32,
        activation='relu'  # Using ReLU activation
    )
    
    nn.train(
        X_train=X_train,
        y_train=y_train_one_hot,
        epochs=100,
        verbose=True,
        adaptive_lr=True
    )
    
    y_test_pred = nn.predict_classes(X_test)
    #save prediction as csv
    output_path = os.path.join(output_folder_path, "prediction_e.csv")
    pd.DataFrame({'prediction':y_test_pred}).to_csv(output_path,index=False) 

def part_f():
   
    clf = MLPClassifier(hidden_layer_sizes=tuple([512,256,128,64]),
                            activation='relu',
                            solver='sgd',
                            learning_rate='invscaling',
                            batch_size=32,
                            alpha=0,
                            max_iter=100,
                            random_state=42,
                            verbose=False)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    output_path = os.path.join(output_folder_path, "prediction_f.csv")
    pd.DataFrame({'prediction':y_pred}).to_csv(output_path,index=False) 

if __name__ == "__main__":
    

    # Check number of arguments
    if len(sys.argv) != 5:
        print("Usage: python decision_tree.py <train_data_path> <test_data_path> <output_folder_path> <question_part>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_folder_path = sys.argv[3]
    part = sys.argv[4].lower()
    
    # Load data
    X_train, y_train, X_test = load_data(train_path, test_path)

    # Question-specific execution
    if part == 'a':
        part_a()

    elif part == 'b':
        part_b()

    elif part == 'c':
        part_c()

    elif part == 'd':
        part_d()

    elif part == 'e':
        part_e()
    
    elif part == 'f':
        part_f()

    else:
        print(f"Invalid question part: {part}. Use one of: b, c, d, e, f.")
