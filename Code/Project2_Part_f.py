#!/usr/bin/env python
# coding: utf-8

# # Part F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def softmax(z):
    """Softmax activation function for multi-class classification"""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def softmax_der(z):
    """Derivative of softmax - will be handled in cross-entropy derivative"""
    # For softmax, the derivative is computed differently in backpropagation
    # We'll handle this in the cross_entropy_der_batch function
    s = softmax(z)
    return s * (1 - s)  # This is simplified, actual derivative is more complex

def cross_entropy(y_pred, y_true):
    """Cross-entropy loss for multi-class classification"""
    # y_pred: predicted probabilities (batch_size, num_classes)
    # y_true: true labels (batch_size, num_classes) in one-hot encoding
    n = y_pred.shape[0]
    # Add small epsilon to avoid log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # For one-hot encoded labels
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        return -np.sum(y_true * np.log(y_pred)) / n
    else:
        # For integer labels
        return -np.sum(np.log(y_pred[np.arange(n), y_true.astype(int)])) / n

def cross_entropy_der_batch(y_pred, y_true):
    """Derivative of cross-entropy loss with softmax"""
    # This combines the derivative of cross-entropy and softmax
    n = y_pred.shape[0]
    
    # If y_true is one-hot encoded
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        return (y_pred - y_true) / n
    else:
        # If y_true is integer labels, convert to one-hot
        y_true_int = y_true.astype(int)
        y_true_one_hot = np.eye(y_pred.shape[1])[y_true_int]
        return (y_pred - y_true_one_hot) / n

def one_hot_encode(y, num_classes=10):
    """Convert integer labels to one-hot encoding"""
    n = y.shape[0]
    y_one_hot = np.zeros((n, num_classes))
    y_one_hot[np.arange(n), y.astype(int)] = 1
    return y_one_hot

def backpropagation_batch_classification(inputs, layers, activation_funcs, targets, activation_ders, 
                                         cost_der=cross_entropy_der_batch):
    N = inputs.shape[0]
    layer_inputs, zs, predict = feed_forward_saver_batch(inputs, layers, activation_funcs)

    layer_grads = [None] * len(layers)
    dC_dz_next = None  

    for i in reversed(range(len(layers))):
        layer_input = layer_inputs[i]      
        z = zs[i]                        
        activation_der = activation_ders[i]

        if i == len(layers) - 1:
            dC_da = cost_der(predict, targets)               
            dC_dz = dC_da
        else:
            W_next, _ = layers[i + 1]                        
            dC_da = dC_dz_next @ W_next                      
            dC_dz = dC_da * activation_der(z).T                  

        dC_dW = dC_dz.T @ layer_input                    
        dC_db = np.sum(dC_dz.T, axis=1, keepdims=True)   

        layer_grads[i] = (dC_dW, dC_db)
        dC_dz_next = dC_dz                                

    return layer_grads

def train_network_mnist(inputs, targets, layers, activation_funcs, activation_ders, lr=0.5, beta1=0.9, beta2=0.999, 
                        eps=1e-8, epochs=100, verbose_every=10, num_classes=10):
    if len(targets.shape) == 1 or targets.shape[1] == 1:
        targets_one_hot = one_hot_encode(targets, num_classes)
    else:
        targets_one_hot = targets
    
    m = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    v = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    
    loss_hist = []
    accuracy_hist = []
    
    for ep in range(epochs):
        preds = feed_forward_batch(inputs, layers, activation_funcs)
        
        loss = cross_entropy(preds, targets_one_hot)
        predicted_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(targets_one_hot, axis=1) if len(targets_one_hot.shape) > 1 else targets.astype(int)
        accuracy = np.mean(predicted_classes == true_classes)
        
        loss_hist.append(loss)
        accuracy_hist.append(accuracy)
        
        grads = backpropagation_batch_classification(
            inputs, layers, activation_funcs, targets_one_hot, activation_ders
        )
        
        new_layers = []
        t = ep + 1
        
        for i, ((W, b), (dW, db), (mW, mb), (vW, vb)) in enumerate(zip(layers, grads, m, v)):
            mW = beta1 * mW + (1 - beta1) * dW
            mb = beta1 * mb + (1 - beta1) * db
            
            vW = beta2 * vW + (1 - beta2) * (dW ** 2)
            vb = beta2 * vb + (1 - beta2) * (db ** 2)
            
            mW_hat = mW / (1 - beta1 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            
            W = W - lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b = b - lr * mb_hat / (np.sqrt(vb_hat) + eps)
            
            new_layers.append((W, b))
            m[i] = (mW, mb)
            v[i] = (vW, vb)
        
        layers = new_layers
        
        if verbose_every and (ep % verbose_every == 0 or ep == epochs-1):
            print(f"Epoch {ep:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    
    return layers, loss_hist, accuracy_hist

def leaky_ReLU(z, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(z > 0, z, alpha * z)

def leaky_ReLU_der(z, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where(z > 0, 1, alpha)

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.full((layer_output_size, 1), 0.01)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = W @ a.T + b
        a = activation_func(z).T
    return a

def feed_forward_saver_batch(inputs, layers, activation_funcs):
    layer_inputs = []   
    zs = []             
    a = inputs

    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = W @ a.T + b                  
        a = activation_func(z).T        
        zs.append(z)
    return layer_inputs, zs, a

def main():
    """Main function to train and evaluate MNIST classifier"""
    # Fetch the MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    # Extract data (features) and target (labels)
    X = mnist.data
    y = mnist.target

    X = X / 255.0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Use smaller training set for faster training
    train_size = 10000  
    X_train_small = X_train_flat[:train_size]
    y_train_small = y_train[:train_size].astype(int)

    X_train_norm = X_train_small / 255.0
    X_test_norm = X_test_flat / 255.0

    # Network architecture
    network_input_size = 784  
    layer_output_sizes = [128, 64, 10]  
    activation_funcs = [leaky_ReLU, leaky_ReLU, softmax]  
    activation_ders = [leaky_ReLU_der, leaky_ReLU_der, softmax_der]

    print(f"Training network architecture: {[network_input_size] + layer_output_sizes}")

    # Initialize layers
    np.random.seed(42)
    layers = create_layers_batch(network_input_size, layer_output_sizes)

    # Train the network
    trained_layers, loss_history, accuracy_history = train_network_mnist(
        X_train_norm, y_train_small, layers, activation_funcs, activation_ders,
        lr=0.001, epochs=50, verbose_every=5
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_preds = feed_forward_batch(X_test_norm, trained_layers, activation_funcs)
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_accuracy = np.mean(test_pred_classes == y_test.astype(int))

    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss (Cross-Entropy)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return trained_layers, loss_history, accuracy_history, test_accuracy

if __name__ == "__main__":
    main()
