#!/usr/bin/env python
# coding: utf-8

# Part E

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def ridge_reg(X, y, lam):
    n_features = X.shape[1]
    I = np.eye(n_features)
    beta = np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y
    return beta

def soft_threshold(z, t):
    """Element-wise soft-threshold"""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def standardize_design_matrix_cols_1p(X):
    """Standardize columns 1..p-1 (leave intercept) for L1 stability."""
    Xs = X.copy()
    if Xs.shape[1] > 1:
        mu = Xs[:, 1:].mean(axis=0)
        sd = Xs[:, 1:].std(axis=0) + 1e-12
        Xs[:, 1:] = (Xs[:, 1:] - mu) / sd
    return Xs

def lasso_ista(X, y, lam, max_iters=10000, tol=1e-8, theta0=None, standardize_cols=True):
    """
    Solve:  C(θ) = (1/n)||y - Xθ||^2 + λ||θ||_1  (no penalty on θ0)
    """
    n, p = X.shape
    
    Xs = standardize_design_matrix_cols_1p(X) if standardize_cols else X
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    
    # Lipschitz constant of the smooth part
    L = (2.0 / n) * np.linalg.eigvalsh(Xs.T @ Xs).max()
    t = 1.0 / (L + 1e-18)
    
    history = {"obj": []}
    
    def obj(theta):
        r = y - Xs @ theta
        return (r @ r) / n + lam * np.sum(np.abs(theta[1:]))
    
    prev = obj(theta)
    history["obj"].append(prev)
    
    for _ in range(max_iters):
        g = (2.0 / n) * (Xs.T @ (Xs @ theta - y))
        z = theta - t * g
        
        z0 = z[0]
        z_rest = soft_threshold(z[1:], lam * t)
        
        theta_new = np.empty_like(theta)
        theta_new[0] = z0
        theta_new[1:] = z_rest
        
        cur = obj(theta_new)
        history["obj"].append(cur)
        
        if abs(prev - cur) < tol:
            theta = theta_new
            break
        
        theta, prev = theta_new, cur
    
    return theta, history, Xs

def standardize_train_test(Xtr, Xte):
    """Standardize train and test design matrices consistently"""
    Xtr_ = Xtr.copy()
    Xte_ = Xte.copy()
    
    if Xtr_.shape[1] > 1:
        mu = Xtr_[:, 1:].mean(axis=0)
        sd = Xtr_[:, 1:].std(axis=0) + 1e-12
        Xtr_[:, 1:] = (Xtr_[:, 1:] - mu) / sd
        Xte_[:, 1:] = (Xte_[:, 1:] - mu) / sd
    
    return Xtr_, Xte_

def lasso_grid_search(degrees, lambdas):
    """Find best MSE across degrees and lambdas"""
    best_mse_ista = float('inf')
    best_params_ista = {}
    
    best_mse_skl = float('inf')
    best_params_skl = {}
    
    for i, d in enumerate(degrees):
        print(f"Processing degree {d}/{degrees[-1]}...")
        
        # Build design matrices
        Xtr = design_matrix_ols(x_train_scaled, d)
        Xte = design_matrix_ols(x_test_scaled, d)
        Xtr_s, Xte_s = standardize_train_test(Xtr, Xte)
        
        for j, lam in enumerate(lambdas):
            # --- ISTA ---
            theta, _, _ = lasso_ista(Xtr_s, y_train, lam=lam, max_iters=4000, 
                                     tol=1e-8, standardize_cols=False)
            yhat_te = Xte_s @ theta
            mse_ista = mse(y_test, yhat_te)
            
            if mse_ista < best_mse_ista:
                best_mse_ista = mse_ista
                best_params_ista = {'degree': d, 'lambda': lam, 'theta': theta}
            
            # --- Scikit-learn ---
            alpha = lam / 2.0
            clf = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-8)
            clf.fit(Xtr_s, y_train)
            yhat_te_skl = clf.predict(Xte_s)
            mse_skl = mse(y_test, yhat_te_skl)
            
            if mse_skl < best_mse_skl:
                best_mse_skl = mse_skl
                best_params_skl = {'degree': d, 'lambda': lam, 'coef': clf.coef_}
    
    return {
        'ista': {'best_mse': best_mse_ista, 'params': best_params_ista},
        'skl': {'best_mse': best_mse_skl, 'params': best_params_skl}
    }

def train_network_sgd_rmsprop_regularization(
    inputs, targets, layers, activation_funcs, activation_ders,
    lr=0.001, decay_rate=0.9, eps=1e-8, epochs=300, verbose_every=25, 
    loss_func=None, l1_lambda=0.0, l2_lambda=0.0
):

    if loss_func is None:
        loss_func = mse
    
    cache = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers]
    loss_hist = []
    
    for ep in range(epochs):
        preds = feed_forward_batch(inputs, layers, activation_funcs)
        
        base_loss = loss_func(preds, targets)
        
        l1_penalty = 0.0
        l2_penalty = 0.0
        if l1_lambda > 0 or l2_lambda > 0:
            for W, b in layers:
                if l1_lambda > 0:
                    l1_penalty += np.sum(np.abs(W))
                if l2_lambda > 0:
                    l2_penalty += np.sum(W ** 2)
        
        total_loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty
        loss_hist.append(total_loss)
        
        grads = backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders)
        
        new_layers = []
        for (W, b), (dW, db), (cW, cb) in zip(layers, grads, cache):
            dW_reg = dW.copy()
            if l1_lambda > 0:
                dW_reg += l1_lambda * np.sign(W)  
            if l2_lambda > 0:
                dW_reg += 2 * l2_lambda * W       
            
            cW = decay_rate * cW + (1 - decay_rate) * (dW_reg ** 2)
            cb = decay_rate * cb + (1 - decay_rate) * (db ** 2)
            
            W = W - lr * dW_reg / (np.sqrt(cW) + eps)
            b = b - lr * db / (np.sqrt(cb) + eps)
            
            new_layers.append((W, b))
            cache[len(new_layers)-1] = (cW, cb)
        layers = new_layers
        
        if verbose_every and (ep % verbose_every == 0 or ep == epochs-1):
            reg_str = ""
            if l1_lambda > 0 or l2_lambda > 0:
                reg_str = f" | L1: {l1_penalty:.4f} | L2: {l2_penalty:.4f}"
            print(f"epoch {ep:4d} | loss {total_loss:.4f} (base: {base_loss:.4f}){reg_str}")
    
    return layers, loss_hist

def main():
    """Main execution function for Part E"""
    np.random.seed(42)

    n_points = 100

    x = np.linspace(-1, 1, n_points).reshape(-1, 1)

    d = 9

    lam = 0.01

    y_true = runge_function(x).ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    X_train = design_matrix_ols(x_train_scaled, d)

    X_test = design_matrix_ols(x_test_scaled, d)

    beta = ridge_reg(X_train, y_train, lam)

    y_train_pred = predict_ols(X_train, beta)

    y_test_pred = predict_ols(X_test, beta)

    print(mse(y_test, y_test_pred))

    np.random.seed(42)

    n_points = 100

    x = np.linspace(-1, 1, n_points).reshape(-1, 1)

    d = 9

    lam = 0.01

    y_true = runge_function(x).ravel()

    y = y_true

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    degrees = np.arange(1, 16)

    lambdas = np.logspace(-4, 1, 30)

    results = lasso_grid_search(degrees, lambdas)

    print("\n*** ISTA (Custom Implementation) ***")

    print(f"Best Test MSE: {results['ista']['best_mse']:.8f}")

    print(f"Optimal Degree: {results['ista']['params']['degree']}")

    print(f"Optimal Lambda: {results['ista']['params']['lambda']:.6f}")

    print(f"Number of non-zero coefficients: {np.sum(np.abs(results['ista']['params']['theta']) > 1e-6)}")

    print("\n*** Scikit-learn LASSO ***")

    print(f"Best Test MSE: {results['skl']['best_mse']:.8f}")

    print(f"Optimal Degree: {results['skl']['params']['degree']}")

    print(f"Optimal Lambda: {results['skl']['params']['lambda']:.6f}")

    print(f"Number of non-zero coefficients: {np.sum(np.abs(results['skl']['params']['coef']) > 1e-6)}")

    np.random.seed(42)

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    targets = y_train.reshape(-1, 1)

    inputs = x_train_scaled

    network_input_size = 1

    layer_output_sizes = [25, 25, 1]

    activation_funcs = [leaky_ReLU, sigmoid, sigmoid]

    activation_ders = [leaky_ReLU_der, sigmoid_der, sigmoid_der]

    l2_lambdas = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    test_mses = []

    for idx, l2_lambda in enumerate(l2_lambdas):
        print(f"\nTesting L2 lambda = {l2_lambda}")

        np.random.seed(42 + idx)

        layers = create_layers_batch(network_input_size, layer_output_sizes)

        layers, _ = train_network_sgd_rmsprop_regularization(
            inputs, targets, layers,
            activation_funcs, activation_ders,
            lr=0.001, decay_rate=0.9, eps=1e-8,
            epochs=epochs, verbose_every=None,
            l1_lambda=0.0,      # No L1
            l2_lambda=l2_lambda  # Current L2 strength
        )

        y_pred_test = feed_forward_batch(x_test_scaled, layers, activation_funcs)
        test_mse = mse(y_test.reshape(-1, 1), y_pred_test)
        test_mses.append(test_mse)

        print(f"  Test MSE: {test_mse:.6f}")

    plt.figure(figsize=(12, 6))

    plt.plot(l2_lambdas, test_mses, 'o-', linewidth=2.5, markersize=8, color='steelblue')

    plt.xlabel('L2 Regularization Strength (λ)', fontsize=14)

    plt.ylabel('Test MSE', fontsize=14)

    plt.title('Effect of L2 Regularization on Test Error', fontsize=16)

    plt.xscale('log')

    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    plt.show()

    np.random.seed(42)

    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.transform(x_test)

    targets = y_train.reshape(-1, 1)

    inputs = x_train_scaled

    network_input_size = 1

    layer_output_sizes = [25, 25, 1]

    activation_funcs = [leaky_ReLU, sigmoid, sigmoid]

    activation_ders = [leaky_ReLU_der, sigmoid_der, sigmoid_der]

    l1_lambdas = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    test_mses = []

    for idx, l1_lambda in enumerate(l1_lambdas):
        print(f"\nTesting L1 lambda = {l1_lambda}")

        np.random.seed(42 + idx)

        layers = create_layers_batch(network_input_size, layer_output_sizes)

        layers, _ = train_network_sgd_rmsprop_regularization(
            inputs, targets, layers,
            activation_funcs, activation_ders,
            lr=0.001, decay_rate=0.9, eps=1e-8,
            epochs=epochs, verbose_every=None,
            l1_lambda=l1_lambda,      # No L1
            l2_lambda=0  # Current L2 strength
        )

        y_pred_test = feed_forward_batch(x_test_scaled, layers, activation_funcs)
        test_mse = mse(y_test.reshape(-1, 1), y_pred_test)
        test_mses.append(test_mse)

        print(f"  Test MSE: {test_mse:.6f}")

    plt.figure(figsize=(12, 6))

    plt.plot(l1_lambdas, test_mses, 'o-', linewidth=2.5, markersize=8, color='steelblue')

    plt.xlabel('L1 Regularization Strength (λ)', fontsize=14)

    plt.ylabel('Test MSE', fontsize=14)

    plt.title('Effect of L1 Regularization on Test Error', fontsize=16)

    plt.xscale('log')

    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
