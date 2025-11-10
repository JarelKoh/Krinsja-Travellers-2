#!/usr/bin/env python
# coding: utf-8

# Part B

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from sklearn import datasets
from sklearn.metrics import accuracy_score

def runge_function(x):
    return 1/ (1 + 25 * x**2)

def runge_noisy(x, noise = 0.1):
    return runge_function(x) + np.random.normal(0, noise, size = x.shape)

def design_matrix_ols(x, degree):
    X = np.ones((x.shape[0], 1))
    for d in range(1, degree+1):
        X = np.hstack((X, x**d))
    return X

def ols_fit(X, y):
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta

def predict_ols(X, beta):
    return X @ beta

def mse(y_true, y_pred):
    n = y_true.shape[0]
    return 0.5 * np.sum((y_pred - y_true)**2) / n

def adam_ols_minibatch_v2(
    X, y,
    alpha=1e-2,           
    beta1=0.9, beta2=0.999, eps=1e-8,
    batch_size=32,
    epochs=50,
    grad_tol=1e-8,
    theta0=None,
    weight_decay=0.0,    
    seed=None
):
    n, p = X.shape
    rng = np.random.default_rng(seed)
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0

    history = {"loss": [], "grad_norm": []}
    steps_per_epoch = math.ceil(n / max(1, batch_size))

    for _ in range(epochs):
        perm = rng.permutation(n) #shuffle once per epoch

        for s in range(steps_per_epoch):
            start = s * batch_size
            end   = min((s + 1) * batch_size, n)
            idx   = perm[start:end]

            Xb, yb = X[idx], y[idx]
            rb   = Xb @ theta - yb
            grad = (2.0 / len(idx)) * (Xb.T @ rb)

            if weight_decay:
                theta *= (1 - alpha * weight_decay)

            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            mhat = m / (1 - beta1**t)
            vhat = v / (1 - beta2**t)
            theta -= alpha * mhat / (np.sqrt(vhat) + eps)

        r_full = X @ theta - y
        loss_full = (r_full @ r_full) / n
        grad_full = (2.0 / n) * (X.T @ r_full)
        gnorm = np.linalg.norm(grad_full)
        history["loss"].append(loss_full)
        history["grad_norm"].append(gnorm)
        if gnorm <= grad_tol:
            break

    return theta, history

def ReLU(z):
    return np.where(z > 0, z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1 - s)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def mse(y_true, y_pred):
    n = y_true.shape[0]
    return 0.5 * np.sum((y_pred - y_true)**2) / n

def mse_der_batch(y, t):
    return (1.0 / y.shape[0]) * (y - t)

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

def backpropagation_batch(
    inputs, layers, activation_funcs, targets, activation_ders, cost_der=mse_der_batch
):
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
        else:
            W_next, _ = layers[i + 1]                        
            dC_da = dC_dz_next @ W_next                      

        dC_dz = dC_da * activation_der(z).T                  

     
        dC_dW = dC_dz.T @ layer_input                    
        dC_db = np.sum(dC_dz.T, axis=1, keepdims=True)   

        layer_grads[i] = (dC_dW, dC_db)

        dC_dz_next = dC_dz                                

    return layer_grads

def train_network_sgd_momentum(
    inputs, targets, layers, activation_funcs, activation_ders,
    lr=0.05, momentum=0.9, epochs=300, verbose_every=25, loss_func=None
):
    if loss_func is None:
        loss_func = mse
    
    velocities = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    loss_hist = []
    
    for ep in range(epochs):
        # forward
        preds = feed_forward_batch(inputs, layers, activation_funcs)
        loss = loss_func(preds, targets)
        loss_hist.append(loss)
        
        grads = backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders)
        new_layers = []
        for (W, b), (dW, db), (vW, vb) in zip(layers, grads, velocities):
            vW = momentum * vW - lr * dW
            vb = momentum * vb - lr * db
            W  = W + vW
            b  = b + vb
            new_layers.append((W, b))
            # write back velocities
            velocities[len(new_layers)-1] = (vW, vb)
        layers = new_layers
        
        if verbose_every and (ep % verbose_every == 0 or ep == epochs-1):
            print(f"epoch {ep:4d} | loss {loss:.4f}")
    
    return layers, loss_hist

def train_network_gd(
    inputs, targets, layers, activation_funcs, activation_ders,
    lr=0.05, epochs=300, verbose_every=25, loss_func=None
):
    if loss_func is None:
        loss_func = mse
    
    loss_hist = []
    
    for ep in range(epochs):
        preds = feed_forward_batch(inputs, layers, activation_funcs)
        loss = loss_func(preds, targets)
        loss_hist.append(loss)
        
        grads = backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders)
        
        new_layers = []
        for (W, b), (dW, db) in zip(layers, grads):
            W = W - lr * dW
            b = b - lr * db
            new_layers.append((W, b))
        layers = new_layers
        
        if verbose_every and (ep % verbose_every == 0 or ep == epochs-1):
            print(f"epoch {ep:4d} | loss {loss:.4f}")
    
    return layers, loss_hist

def train_network_sgd_rmsprop(
    inputs, targets, layers, activation_funcs, activation_ders,
    lr=0.001, decay_rate=0.9, eps=1e-8, epochs=300, verbose_every=25, loss_func=None
):
    if loss_func is None:
        loss_func = mse
    
    cache = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    loss_hist = []
    
    for ep in range(epochs):
        # forward
        preds = feed_forward_batch(inputs, layers, activation_funcs)
        loss = loss_func(preds, targets)
        loss_hist.append(loss)
        

        grads = backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders)
        
        new_layers = []
        for (W, b), (dW, db), (cW, cb) in zip(layers, grads, cache):
            cW = decay_rate * cW + (1 - decay_rate) * (dW ** 2)
            cb = decay_rate * cb + (1 - decay_rate) * (db ** 2)
            
            W = W - lr * dW / (np.sqrt(cW) + eps)
            b = b - lr * db / (np.sqrt(cb) + eps)
            
            new_layers.append((W, b))
            cache[len(new_layers)-1] = (cW, cb)
        layers = new_layers
        
        if verbose_every and (ep % verbose_every == 0 or ep == epochs-1):
            print(f"epoch {ep:4d} | loss {loss:.4f}")
    
    return layers, loss_hist

def train_network_sgd_adam(
    inputs, targets, layers, activation_funcs, activation_ders,
    lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, epochs=300, verbose_every=25, loss_func=None
):
    if loss_func is None:
        loss_func = mse
    
    # Initialize first and second moment estimates
    m = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    v = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    loss_hist = []
    
    for ep in range(epochs):
        # forward
        preds = feed_forward_batch(inputs, layers, activation_funcs)
        loss = loss_func(preds, targets)
        loss_hist.append(loss)
        
        # backward
        grads = backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders)
        
        # Adam update
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
            print(f"epoch {ep:4d} | loss {loss:.4f}")
    
    return layers, loss_hist

def runge_function_2d(x, y):
    return 1 / ((10*x - 5)**2 + (10*y - 5)**2 + 1)

def main():
    """Main execution function for Part B"""
    np.random.seed(42)

    n_points = 100

    x = np.linspace(-1, 1, n_points).reshape(-1, 1)

    y_true = runge_function(x).ravel()

    y = runge_noisy(x, noise = 0.1).ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    degrees = range(1, 24)

    mse_train, mse_test = [], []

    for d in degrees:
        X_train = design_matrix_ols(x_train_scaled, d)
        X_test = design_matrix_ols(x_test_scaled, d)

        beta = ols_fit(X_train, y_train)

        y_train_pred = predict_ols(X_train, beta)
        y_test_pred = predict_ols(X_test, beta)

        mse_train.append(mse(y_train, y_train_pred))
        mse_test.append(mse(y_test, y_test_pred))

    plt.figure(figsize=(8, 5))

    plt.plot(degrees, mse_train, 'o-', label="Train MSE")

    plt.plot(degrees, mse_test, 'o-', label="Test MSE")

    plt.xlabel("Polynomial Degree")

    plt.ylabel("MSE")

    plt.legend()

    plt.title("MSE vs Polynomial Degree")

    plt.show()

    X_train = design_matrix_ols(x_train_scaled, 9)

    X_test = design_matrix_ols(x_test_scaled, 9)

    beta = ols_fit(X_train, y_train)

    y_train_pred = predict_ols(X_train, beta)

    y_test_pred = predict_ols(X_test, beta)

    test_mse = mse(y_test_pred, y_test)

    print(f"Test MSE for OLS: {test_mse:.4f}")

    theta0  = np.zeros(X_train.shape[1])

    beta, _ = adam_ols_minibatch_v2(X_train, y_train, theta0)

    ytr_hat = predict_ols(X_train, beta)

    yte_hat = predict_ols(X_test,  beta)

    test_mse = mse(y_test, yte_hat)

    print(f"Test MSE for Stochastic Gradient Descent with ADAM: {test_mse:.4f}")

    np.random.seed(42)

    X_train = design_matrix_ols(x_train_scaled, 9)

    X_test = design_matrix_ols(x_test_scaled, 9)

    beta = ols_fit(X_train, y_train)

    y_train_pred = predict_ols(X_train, beta)

    y_test_pred = predict_ols(X_test, beta)

    test_mse_ols = mse(y_test_pred, y_test)

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    targets = y_train.reshape(-1, 1)

    inputs = x_train_scaled

    network_input_size = 1

    layer_output_sizes = [50, 1]

    activation_funcs = [sigmoid, sigmoid]

    activation_ders = [sigmoid_der, sigmoid_der]

    layers = create_layers_batch(network_input_size, layer_output_sizes)

    layers = create_layers_batch(network_input_size, layer_output_sizes)

    layers,loss_hist = train_network_sgd_adam(
        inputs, targets, layers, activation_funcs, activation_ders,
        lr=0.001, beta1=0.9, beta2=0.999, epochs=500, verbose_every=50
    )

    y_pred_test = feed_forward_batch(x_test_scaled, layers, activation_funcs)

    test_mse = mse(y_test.reshape(-1, 1), y_pred_test)

    layer_output_sizes = [50, 100, 1]

    activation_funcs = [sigmoid, sigmoid, sigmoid]

    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der]

    layers = create_layers_batch(network_input_size, layer_output_sizes)

    layers = create_layers_batch(network_input_size, layer_output_sizes)

    layers,loss_hist = train_network_sgd_adam(
        inputs, targets, layers, activation_funcs, activation_ders,
        lr=0.001, beta1=0.9, beta2=0.999, epochs=500, verbose_every=50
    )

    y_pred_test = feed_forward_batch(x_test_scaled, layers, activation_funcs)

    test_mse = mse(y_test.reshape(-1, 1), y_pred_test)

    print(f"\nTest MSE for Neural Network with one hidden layer: {test_mse:.4f}")

    print(f"\nTest MSE for Neural Network with two hidden layers: {test_mse:.4f}")

    print(f"Test MSE for OLS: {test_mse_ols:.4f}")

    np.random.seed(42)

    n_points = 20

    x_1d = np.linspace(0, 1, n_points)

    y_1d = np.linspace(0, 1, n_points)

    x_mesh, y_mesh = np.meshgrid(x_1d, y_1d)

    x_flat = x_mesh.ravel()

    y_flat = y_mesh.ravel()

    X = np.column_stack([x_flat, y_flat])

    z_true = runge_function_2d(x_flat, y_flat)

    X_train, X_test, z_train, z_test = train_test_split(
        X, z_true, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {X_train_scaled.shape[0]}")

    print(f"Test samples: {X_test_scaled.shape[0]}")

    print(f"Input features: {X_train_scaled.shape[1]}")

    targets = z_train.reshape(-1, 1)

    inputs = X_train_scaled

    network_input_size = 2

    layer_output_sizes = [50,100, 1]

    activation_funcs = [sigmoid, sigmoid, sigmoid]

    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der]

    layers = create_layers_batch(network_input_size, layer_output_sizes)

    layers, loss_hist = train_network_sgd_adam(
        inputs, targets, layers, activation_funcs, activation_ders,
        lr=0.001, beta1=0.9, beta2=0.999, epochs=500, verbose_every=50
    )

    z_pred_test = feed_forward_batch(X_test_scaled, layers, activation_funcs)

    test_mse = mse(z_test.reshape(-1, 1), z_pred_test)

    z_pred_train = feed_forward_batch(X_train_scaled, layers, activation_funcs)

    train_mse = mse(z_train.reshape(-1, 1), z_pred_train)

    print(f"\nTrain MSE: {train_mse:.6f}")

    print(f"Test MSE: {test_mse:.6f}")

    plt.figure(figsize=(10, 5))

    plt.plot(loss_hist)

    plt.xlabel('Epoch')

    plt.ylabel('Training Loss (MSE)')

    plt.title('Training Loss Over Time - 2D Runge Function')

    plt.grid(True, alpha=0.3)

    plt.show()

    np.random.seed(42)

    inputs = X_train_scaled

    targets = z_train.reshape(-1, 1)

    network_input_size = 2

    layer_output_sizes = [50, 100, 1]

    activation_funcs = [sigmoid, sigmoid, sigmoid]

    activation_ders = [sigmoid_der, sigmoid_der, sigmoid_der]

    optimizer_configs = {
        'Plain GD': {
            'function': train_network_gd,
            'learning_rates': [0.001, 0.01, 0.05, 0.1, 0.5],
            'params': {}
        },
        'SGD Momentum': {
            'function': train_network_sgd_momentum,
            'learning_rates': [0.001, 0.01, 0.05, 0.1, 0.5],
            'params': {'momentum': 0.9}
        },
        'RMSprop': {
            'function': train_network_sgd_rmsprop,
            'learning_rates': [0.0001, 0.001, 0.01, 0.1],
            'params': {'decay_rate': 0.9}
        },
        'Adam': {
            'function': train_network_sgd_adam,
            'learning_rates': [0.0001, 0.001, 0.01, 0.1],
            'params': {'beta1': 0.9, 'beta2': 0.999}
        }
    }

    all_results = {}

    for opt_name, config in optimizer_configs.items():
        print(f"\nTesting {opt_name}:")
        print("=" * 60)

        opt_results = {}

        for lr in config['learning_rates']:
            np.random.seed(42)
            layers = create_layers_batch(network_input_size, layer_output_sizes)


            layers, loss_hist = config['function'](
                inputs, targets, layers, activation_funcs, activation_ders,
                lr=lr, epochs=500, verbose_every=None, **config['params']
            )


            z_pred_test = feed_forward_batch(X_test_scaled, layers, activation_funcs)  # Changed variable names
            test_mse = mse(z_test.reshape(-1, 1), z_pred_test)  # Changed from y_test


            z_pred_train = feed_forward_batch(X_train_scaled, layers, activation_funcs)  # Changed variable names
            train_mse = mse(z_train.reshape(-1, 1), z_pred_train)  # Changed from y_train

            opt_results[lr] = {
                'test_mse': test_mse,
                'train_mse': train_mse,
                'loss_hist': loss_hist
            }

            print(f"  lr={lr:.4f} | Train: {train_mse:.6f} | Test: {test_mse:.6f}")


        best_lr = min(opt_results.keys(), key=lambda lr: opt_results[lr]['test_mse'])
        print(f"  Best lr for {opt_name}: {best_lr:.4f} (Test MSE: {opt_results[best_lr]['test_mse']:.6f})")

        all_results[opt_name] = opt_results

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes = axes.ravel()

    for idx, (opt_name, opt_results) in enumerate(all_results.items()):
        ax = axes[idx]

        lrs = list(opt_results.keys())
        test_mses = [opt_results[lr]['test_mse'] for lr in lrs]
        train_mses = [opt_results[lr]['train_mse'] for lr in lrs]

        best_lr = min(lrs, key=lambda lr: opt_results[lr]['test_mse'])

        ax.plot(lrs, test_mses, 'o-', label='Test MSE', linewidth=2, markersize=8)
        ax.plot(lrs, train_mses, 's-', label='Train MSE', linewidth=2, markersize=8)
        ax.axvline(best_lr, color='red', linestyle='--', alpha=0.5, label=f'Best lr={best_lr}')

        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(f'{opt_name} - MSE vs Learning Rate (2D Runge)', fontsize=14)
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()

    print("\n" + "=" * 80)

    print("SUMMARY: Best Learning Rate for Each Optimizer (2D Runge Function)")

    print("=" * 80)

    print(f"{'Optimizer':<20} {'Best LR':<15} {'Test MSE':<15} {'Train MSE':<15}")

    print("=" * 80)

    for opt_name, opt_results in all_results.items():
        best_lr = min(opt_results.keys(), key=lambda lr: opt_results[lr]['test_mse'])
        test_mse = opt_results[best_lr]['test_mse']
        train_mse = opt_results[best_lr]['train_mse']
        print(f"{opt_name:<20} {best_lr:<15.4f} {test_mse:<15.6f} {train_mse:<15.6f}")

    best_overall = min(
        [(opt, lr, res['test_mse']) 
         for opt, opt_res in all_results.items() 
         for lr, res in opt_res.items()],
        key=lambda x: x[2]
    )

    print(f"\nOverall Best: {best_overall[0]} with lr={best_overall[1]:.4f}, Test MSE={best_overall[2]:.6f}")

    best_opt, best_lr_overall, _ = best_overall

    best_layers = None

    np.random.seed(42)

    layers = create_layers_batch(network_input_size, layer_output_sizes)

    layers, _ = optimizer_configs[best_opt]['function'](
        inputs, targets, layers, activation_funcs, activation_ders,
        lr=best_lr_overall, epochs=500, verbose_every=None,
        **optimizer_configs[best_opt]['params']
    )

    ax3 = fig.add_subplot(133, projection='3d')

    error_mesh = np.abs(z_true_mesh - z_pred_mesh)

    surf3 = ax3.plot_surface(x_mesh, y_mesh, error_mesh, cmap='Reds', alpha=0.8)

    ax3.set_xlabel('x')

    ax3.set_ylabel('y')

    ax3.set_zlabel('|Error|')

    ax3.set_title('Absolute Prediction Error')

    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
