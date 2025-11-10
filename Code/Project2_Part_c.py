#!/usr/bin/env python
# coding: utf-8

# Part C

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, regularizers
from mpl_toolkits.mplot3d import Axes3D

def create_neural_network_keras_2d(n_neurons_layer1, n_neurons_layer2, n_outputs, eta, lmbd):
    model = Sequential()
    
    model.add(Dense(n_neurons_layer1, activation='sigmoid', 
                   kernel_regularizer=regularizers.l2(lmbd),
                   input_shape=(2,)))  

    model.add(Dense(n_neurons_layer2, activation='sigmoid', 
                   kernel_regularizer=regularizers.l2(lmbd)))
    
    model.add(Dense(n_outputs, activation='sigmoid'))
    

    rmsprop = optimizers.RMSprop(learning_rate=eta, rho=0.9)  
    
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mse'])
    
    return model

def main():
    """Main execution function for Part C"""
    np.random.seed(42)

    tf.random.set_seed(42)

    epochs = 500

    batch_size = 32

    n_neurons_layer1 = 50

    n_neurons_layer2 = 100

    n_outputs = 1

    eta_vals = np.array([0.0001, 0.001, 0.01, 0.1])

    lmbd = 0.0

    X_train = X_train_scaled

    Y_train = z_train.reshape(-1, 1)

    X_test = X_test_scaled

    Y_test = z_test.reshape(-1, 1)

    results_keras_2d = {}

    for eta in eta_vals:
        print(f"\nLearning rate = {eta}")
        print("-" * 70)


        np.random.seed(42)
        tf.random.set_seed(42)


        DNN = create_neural_network_keras_2d(n_neurons_layer1, n_neurons_layer2, n_outputs,
                                            eta=eta, lmbd=lmbd)


        history = DNN.fit(X_train, Y_train, 
                         epochs=epochs, 
                         batch_size=batch_size, 
                         verbose=0,
                         validation_data=(X_test, Y_test))


        test_loss, test_mse = DNN.evaluate(X_test, Y_test, verbose=0)


        train_loss, train_mse = DNN.evaluate(X_train, Y_train, verbose=0)


        results_keras_2d[eta] = {
            'model': DNN,
            'history': history,
            'train_mse': train_mse,
            'test_mse': test_mse
        }

        print(f"Train MSE: {train_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")

    print("\n" + "=" * 70)

    print("SUMMARY: Keras 2D Results (RMSprop)")

    print("=" * 70)

    print(f"{'Learning Rate':<15} {'Train MSE':<15} {'Test MSE':<15}")

    print("=" * 70)

    for eta in eta_vals:
        train_mse = results_keras_2d[eta]['train_mse']
        test_mse = results_keras_2d[eta]['test_mse']
        print(f"{eta:<15.4f} {train_mse:<15.6f} {test_mse:<15.6f}")

    best_eta_2d = min(results_keras_2d.keys(), key=lambda k: results_keras_2d[k]['test_mse'])

    best_test_mse_2d = results_keras_2d[best_eta_2d]['test_mse']

    print("=" * 70)

    print(f"Best Learning Rate: {best_eta_2d:.4f}")

    print(f"Best Test MSE: {best_test_mse_2d:.6f}")

    print("=" * 70)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)

    for eta in eta_vals:
        history = results_keras_2d[eta]['history']
        plt.plot(history.history['loss'], label=f'lr={eta:.4f}', alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)

    plt.ylabel('Training Loss (MSE)', fontsize=12)

    plt.title('Keras RMSprop - Training Loss vs Epoch', fontsize=14)

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)

    for eta in eta_vals:
        history = results_keras_2d[eta]['history']
        plt.plot(history.history['val_loss'], label=f'lr={eta:.4f}', alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)

    plt.ylabel('Validation Loss (MSE)', fontsize=12)

    plt.title('Keras RMSprop - Validation Loss vs Epoch', fontsize=14)

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

    ax3 = fig.add_subplot(133, projection='3d')

    error_mesh = np.abs(runge_function_2d(x_mesh, y_mesh) - z_pred_keras)

    ax3.plot_surface(x_mesh, y_mesh, error_mesh, cmap='Reds', alpha=0.8)

    ax3.set_xlabel('x')

    ax3.set_ylabel('y')

    ax3.set_zlabel('|Error|')

    ax3.set_title('Absolute Prediction Error')

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
