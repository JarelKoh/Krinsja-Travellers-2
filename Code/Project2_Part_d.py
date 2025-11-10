#!/usr/bin/env python
# coding: utf-8

# Part D

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def create_activation_functions(num_layers):
    """Leaky ReLU for first layer, Sigmoid for rest"""
    activation_funcs = []
    activation_ders = []
    
    for i in range(num_layers):
        if i == 0:
            activation_funcs.append(leaky_ReLU)
            activation_ders.append(leaky_ReLU_der)
        else:
            activation_funcs.append(sigmoid)
            activation_ders.append(sigmoid_der)
    
    return activation_funcs, activation_ders

def train_and_predict_nn(X_train, y_train, X_test, layer_sizes, epochs=500, lr=0.01,
                        l1_lambda=0.0, l2_lambda=0.0):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    targets = y_train.reshape(-1, 1)
    
    num_layers = len(layer_sizes)
    activation_funcs, activation_ders = create_activation_functions(num_layers)
    layers = create_layers_batch(2, layer_sizes)  # network_input_size = 2
    
    layers, _ = train_network_sgd_rmsprop(
        X_train_scaled, targets, layers,
        activation_funcs, activation_ders,
        lr=lr, decay_rate=0.9, eps=1e-8,
        epochs=epochs, verbose_every=None  # Add L2 regularization parameter
    )
    
    y_pred = feed_forward_batch(X_test_scaled, layers, activation_funcs)
    
    return y_pred.ravel()

def main():
    """Main execution function for Part D"""
    np.random.seed(42)

    targets = z_train.reshape(-1, 1)

    inputs = X_train_scaled

    network_input_size = 2

    layer_output_sizes = [50, 100, 1]

    activation_combos = {
        'All Sigmoid': {
            'funcs': [sigmoid, sigmoid, sigmoid],
            'ders': [sigmoid_der, sigmoid_der, sigmoid_der],
            'description': 'Sigmoid → Sigmoid → Sigmoid'
        },
        'All ReLU': {
            'funcs': [ReLU, ReLU, ReLU],
            'ders': [ReLU_der, ReLU_der, ReLU_der],
            'description': 'ReLU → ReLU → ReLU'
        },
        'All Leaky ReLU': {
            'funcs': [leaky_ReLU, leaky_ReLU, leaky_ReLU],
            'ders': [leaky_ReLU_der, leaky_ReLU_der, leaky_ReLU_der],
            'description': 'Leaky ReLU → Leaky ReLU → Leaky ReLU'
        },
        'ReLU-Sigmoid': {
            'funcs': [ReLU, ReLU, sigmoid],
            'ders': [ReLU_der, ReLU_der, sigmoid_der],
            'description': 'ReLU → ReLU → Sigmoid (output)'
        },
        'Leaky ReLU-Sigmoid': {
            'funcs': [leaky_ReLU, leaky_ReLU, sigmoid],
            'ders': [leaky_ReLU_der, leaky_ReLU_der, sigmoid_der],
            'description': 'Leaky ReLU → Leaky ReLU → Sigmoid (output)'
        },
        'Sigmoid-ReLU': {
            'funcs': [sigmoid, sigmoid, ReLU],
            'ders': [sigmoid_der, sigmoid_der, ReLU_der],
            'description': 'Sigmoid → Sigmoid → ReLU (output)'
        },
        'Mixed: Sigmoid-ReLU-Sigmoid': {
            'funcs': [sigmoid, ReLU, sigmoid],
            'ders': [sigmoid_der, ReLU_der, sigmoid_der],
            'description': 'Sigmoid → ReLU → Sigmoid'
        },
        'Mixed: ReLU-Sigmoid-ReLU': {
            'funcs': [ReLU, sigmoid, ReLU],
            'ders': [ReLU_der, sigmoid_der, ReLU_der],
            'description': 'ReLU → Sigmoid → ReLU'
        },
        'Mixed: ReLU-Leaky-Sigmoid': {
            'funcs': [ReLU, leaky_ReLU, sigmoid],
            'ders': [ReLU_der, leaky_ReLU_der, sigmoid_der],
            'description': 'ReLU → Leaky ReLU → Sigmoid'
        },
        'Mixed: Leaky-Sigmoid-Sigmoid': {
            'funcs': [leaky_ReLU, sigmoid, sigmoid],
            'ders': [leaky_ReLU_der, sigmoid_der, sigmoid_der],
            'description': 'Leaky ReLU → Sigmoid → Sigmoid'
        }
    }

    results_combos = {}

    for combo_name, config in activation_combos.items():
        print(f"\n{combo_name}: {config['description']}")
        print("-" * 80)

        np.random.seed(42)

        layers = create_layers_batch(network_input_size, layer_output_sizes)


        try:
            layers, loss_hist = train_network_sgd_rmsprop(
                inputs, targets, layers, 
                config['funcs'], config['ders'],
                lr=0.01, decay_rate=0.9, eps=1e-8, 
                epochs=500, verbose_every=None
            )


            z_pred_test = feed_forward_batch(X_test_scaled, layers, config['funcs'])
            test_mse = mse(z_test.reshape(-1, 1), z_pred_test)


            z_pred_train = feed_forward_batch(X_train_scaled, layers, config['funcs'])
            train_mse = mse(z_train.reshape(-1, 1), z_pred_train)


            converged = "✓" if test_mse < 0.1 else "✗ (Failed)"


            results_combos[combo_name] = {
                'layers': layers,
                'loss_hist': loss_hist,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'funcs': config['funcs'],
                'description': config['description'],
                'converged': converged
            }

            print(f"Train MSE: {train_mse:.6f}")
            print(f"Test MSE: {test_mse:.6f} {converged}")

        except Exception as e:
            print(f"ERROR: Training failed - {e}")
            results_combos[combo_name] = {
                'train_mse': float('inf'),
                'test_mse': float('inf'),
                'converged': "✗ (Error)",
                'description': config['description']
            }

    sorted_combos = sorted(results_combos.items(), key=lambda x: x[1]['test_mse'])

    for combo_name, result in sorted_combos:
        train_mse = result['train_mse']
        test_mse = result['test_mse']
        status = result['converged']
        description = result['description']


        train_str = f"{train_mse:.6f}" if train_mse != float('inf') else "FAILED"
        test_str = f"{test_mse:.6f}" if test_mse != float('inf') else "FAILED"

        print(f"{combo_name:<30} {description:<35} {train_str:<12} {test_str:<12} {status:<10}")

    best_combo = min([k for k in results_combos.keys() if results_combos[k]['test_mse'] != float('inf')],
                    key=lambda k: results_combos[k]['test_mse'])

    best_test_mse = results_combos[best_combo]['test_mse']

    best_description = results_combos[best_combo]['description']

    print(f"Best Combination: {best_combo}")

    print(f"Architecture: {best_description}")

    print(f"Best Test MSE: {best_test_mse:.6f}")

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)

    top_5 = sorted_combos[:5]

    for combo_name, result in top_5:
        if result['test_mse'] != float('inf'):
            plt.plot(result['loss_hist'], label=combo_name, alpha=0.7, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)

    plt.ylabel('Training Loss (MSE)', fontsize=12)

    plt.title('Training Loss - Top 5 Activation Combinations', fontsize=14)

    plt.legend(fontsize=10)

    plt.grid(True, alpha=0.3)

    plt.yscale('log')

    plt.subplot(2, 2, 2)

    top_8 = sorted_combos[:8]

    combo_names = [name for name, _ in top_8]

    train_mses = [results_combos[name]['train_mse'] for name in combo_names]

    test_mses = [results_combos[name]['test_mse'] for name in combo_names]

    x = np.arange(len(combo_names))

    width = 0.35

    plt.bar(x - width/2, train_mses, width, label='Train MSE', alpha=0.8)

    plt.bar(x + width/2, test_mses, width, label='Test MSE', alpha=0.8)

    plt.xlabel('Activation Combination', fontsize=12)

    plt.ylabel('MSE', fontsize=12)

    plt.title('MSE Comparison - Top 8 Combinations', fontsize=14)

    plt.xticks(x, combo_names, rotation=45, ha='right', fontsize=9)

    plt.legend()

    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    np.random.seed(42)

    n_points = 20

    x_1d = np.linspace(0, 1, n_points)

    y_1d = np.linspace(0, 1, n_points)

    x_mesh, y_mesh = np.meshgrid(x_1d, y_1d)

    x_flat = x_mesh.ravel()

    y_flat = y_mesh.ravel()

    X_full = np.column_stack([x_flat, y_flat])

    z_true_full = runge_function_2d(x_flat, y_flat)

    X_data, X_test, z_data, z_test = train_test_split(
        X_full, z_true_full, test_size=0.3, random_state=42
    )

    n_bootstraps = 50

    hidden_layer_range = range(1, 16)

    results = {
        'n_layers': [],
        'bias_squared': [],
        'variance': [],
        'error': []
    }

    for n_hidden in hidden_layer_range:
        print(f"\nTesting {n_hidden} hidden layer(s)...")

        layer_sizes = [50] * n_hidden + [1]

        bootstrap_predictions = np.zeros((n_bootstraps, len(z_test)))

        successful_bootstraps = 0
        for i in range(n_bootstraps):
            if (i + 1) % 20 == 0:
                print(f"  Bootstrap {i + 1}/{n_bootstraps}")


            X_bootstrap, z_bootstrap = resample(X_data, z_data, random_state=i)


            try:
                y_pred = train_and_predict_nn(
                    X_bootstrap, z_bootstrap, X_test, 
                    layer_sizes, epochs=500, lr=0.01
                )
                bootstrap_predictions[i, :] = y_pred
                successful_bootstraps += 1
            except Exception as e:
                bootstrap_predictions[i, :] = np.nan

        print(f"  Completed: {successful_bootstraps}/{n_bootstraps} successful")

        valid_mask = ~np.isnan(bootstrap_predictions).any(axis=1)
        bootstrap_predictions = bootstrap_predictions[valid_mask]



        y_pred_mean = np.mean(bootstrap_predictions, axis=0)

        bias = y_pred_mean - z_test  
        bias_squared = np.mean(bias ** 2)

        variance = np.mean(np.var(bootstrap_predictions, axis=0))

        individual_errors = np.mean((bootstrap_predictions - z_test) ** 2, axis=1)
        error = np.mean(individual_errors) 

        results['n_layers'].append(n_hidden)
        results['bias_squared'].append(bias_squared)
        results['variance'].append(variance)
        results['error'].append(error)

        print(f"  Bias²: {bias_squared:.6f}, Variance: {variance:.6f}, Error: {error:.6f}")

    plt.figure(figsize=(12, 7))

    plt.plot(results['n_layers'], results['error'], 'o-', linewidth=2.5, 
             markersize=8, label='Error (Test MSE)', color='steelblue')

    plt.plot(results['n_layers'], results['bias_squared'], 's-', linewidth=2.5, 
             markersize=8, label='Bias²', color='darkorange')

    plt.plot(results['n_layers'], results['variance'], '^-', linewidth=2.5, 
             markersize=8, label='Variance', color='green')

    plt.xlabel('Number of Hidden Layers', fontsize=14)

    plt.ylabel('Value', fontsize=14)

    plt.title('Bootstrap Bias-Variance Trade-off (2D Runge Function)', fontsize=16)

    plt.legend(fontsize=12, loc='upper left')

    plt.grid(True, alpha=0.3)

    plt.xticks(results['n_layers'])

    all_values = (results['error'] + results['bias_squared'] + results['variance'])

    y_max = max(all_values) * 1.5

    plt.ylim(0, y_max)

    plt.tight_layout()

    plt.show()

    np.random.seed(42)

    n_points = 20

    x_1d = np.linspace(0, 1, n_points)

    y_1d = np.linspace(0, 1, n_points)

    x_mesh, y_mesh = np.meshgrid(x_1d, y_1d)

    x_flat = x_mesh.ravel()

    y_flat = y_mesh.ravel()

    X_full = np.column_stack([x_flat, y_flat])

    z_true_full = runge_function_2d(x_flat, y_flat)

    X_data, X_test, z_data, z_test = train_test_split(
        X_full, z_true_full, test_size=0.3, random_state=42
    )

    n_bootstraps = 50

    n_hidden_layers = 5

    neuron_range = [10, 25, 50, 75, 100, 150, 200, 250, 300]

    results = {
        'n_neurons': [],
        'bias_squared': [],
        'variance': [],
        'error': []
    }

    for n_neurons in neuron_range:
        print(f"\nTesting {n_neurons} neurons per layer...")

        layer_sizes = [n_neurons] * n_hidden_layers + [1]

        bootstrap_predictions = np.zeros((n_bootstraps, len(z_test)))

        successful_bootstraps = 0
        for i in range(n_bootstraps):
            if (i + 1) % 20 == 0:
                print(f"  Bootstrap {i + 1}/{n_bootstraps}")

            X_bootstrap, z_bootstrap = resample(X_data, z_data, random_state=i)

            try:
                y_pred = train_and_predict_nn(
                    X_bootstrap, z_bootstrap, X_test, 
                    layer_sizes, epochs=500, lr=0.01
                )
                bootstrap_predictions[i, :] = y_pred
                successful_bootstraps += 1
            except Exception as e:
                bootstrap_predictions[i, :] = np.nan

        print(f"  Completed: {successful_bootstraps}/{n_bootstraps} successful")

        valid_mask = ~np.isnan(bootstrap_predictions).any(axis=1)
        bootstrap_predictions = bootstrap_predictions[valid_mask]

        y_pred_mean = np.mean(bootstrap_predictions, axis=0)

        bias = y_pred_mean - z_test  
        bias_squared = np.mean(bias ** 2)

        variance = np.mean(np.var(bootstrap_predictions, axis=0))

        individual_errors = np.mean((bootstrap_predictions - z_test) ** 2, axis=1)
        error = np.mean(individual_errors) 

        results['n_neurons'].append(n_neurons)
        results['bias_squared'].append(bias_squared)
        results['variance'].append(variance)
        results['error'].append(error)

        print(f"  Bias²: {bias_squared:.6f}, Variance: {variance:.6f}, Error: {error:.6f}")

    plt.figure(figsize=(12, 7))

    plt.plot(results['n_neurons'], results['error'], 'o-', linewidth=2.5, 
             markersize=8, label='Error (Test MSE)', color='steelblue')

    plt.plot(results['n_neurons'], results['bias_squared'], 's-', linewidth=2.5, 
             markersize=8, label='Bias²', color='darkorange')

    plt.plot(results['n_neurons'], results['variance'], '^-', linewidth=2.5, 
             markersize=8, label='Variance', color='green')

    plt.xlabel('Number of Neurons per Layer', fontsize=14)

    plt.ylabel('Value', fontsize=14)

    plt.title(f'Bootstrap Bias-Variance Trade-off ({n_hidden_layers} Hidden Layers)', fontsize=16)

    plt.legend(fontsize=12, loc='upper right')

    plt.grid(True, alpha=0.3)

    plt.xticks(results['n_neurons'], rotation=45)

    all_values = (results['error'] + results['bias_squared'] + results['variance'])

    y_max = max(all_values) * 1.5

    plt.ylim(0, y_max)

    plt.tight_layout()

    plt.show()

    np.random.seed(42)

    n_points = 20

    x_1d = np.linspace(0, 1, n_points)

    y_1d = np.linspace(0, 1, n_points)

    x_mesh, y_mesh = np.meshgrid(x_1d, y_1d)

    x_flat = x_mesh.ravel()

    y_flat = y_mesh.ravel()

    X_full = np.column_stack([x_flat, y_flat])

    z_true_full = runge_function_2d(x_flat, y_flat)

    X_data, X_test, z_data, z_test = train_test_split(
        X_full, z_true_full, test_size=0.3, random_state=42
    )

    n_bootstraps = 20

    layer_range = [1, 2, 3, 4, 5, 6, 8, 10]

    neuron_range = [10, 25, 50, 75, 100, 150, 200]

    error_matrix = np.zeros((len(neuron_range), len(layer_range)))

    bias_matrix = np.zeros((len(neuron_range), len(layer_range)))

    variance_matrix = np.zeros((len(neuron_range), len(layer_range)))

    total_iterations = len(layer_range) * len(neuron_range)

    current_iteration = 0

    for i, n_neurons in enumerate(neuron_range):
        for j, n_layers in enumerate(layer_range):
            current_iteration += 1
            print(f"\n[{current_iteration}/{total_iterations}] Testing: {n_layers} layers × {n_neurons} neurons")

            layer_sizes = [n_neurons] * n_layers + [1]

            bootstrap_predictions = np.zeros((n_bootstraps, len(z_test)))

            successful_bootstraps = 0
            for k in range(n_bootstraps):
                X_bootstrap, z_bootstrap = resample(X_data, z_data, random_state=k)

                try:
                    y_pred = train_and_predict_nn(
                        X_bootstrap, z_bootstrap, X_test,
                        layer_sizes, epochs=500, lr=0.01
                    )
                    bootstrap_predictions[k, :] = y_pred
                    successful_bootstraps += 1
                except Exception as e:
                    bootstrap_predictions[k, :] = np.nan

            # Remove failed bootstraps
            valid_mask = ~np.isnan(bootstrap_predictions).any(axis=1)
            bootstrap_predictions = bootstrap_predictions[valid_mask]



            y_pred_mean = np.mean(bootstrap_predictions, axis=0)
            bias = y_pred_mean - z_test
            bias_squared = np.mean(bias ** 2)
            variance = np.mean(np.var(bootstrap_predictions, axis=0))
            individual_errors = np.mean((bootstrap_predictions - z_test) ** 2, axis=1)
            error = np.mean(individual_errors)

            error_matrix[i, j] = error
            bias_matrix[i, j] = bias_squared
            variance_matrix[i, j] = variance

            print(f"  Bias²: {bias_squared:.6f}, Variance: {variance:.6f}, Error: {error:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]

    im1 = ax1.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower')

    ax1.set_xlabel('Number of Hidden Layers', fontsize=12)

    ax1.set_ylabel('Number of Neurons per Layer', fontsize=12)

    ax1.set_title('Test Error (MSE)', fontsize=14, fontweight='bold')

    ax1.set_xticks(range(len(layer_range)))

    ax1.set_xticklabels(layer_range)

    ax1.set_yticks(range(len(neuron_range)))

    ax1.set_yticklabels(neuron_range)

    cbar1 = plt.colorbar(im1, ax=ax1)

    cbar1.set_label('MSE', fontsize=11)

    for i in range(len(neuron_range)):
        for j in range(len(layer_range)):
            if not np.isnan(error_matrix[i, j]):
                text = ax1.text(j, i, f'{error_matrix[i, j]:.4f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if error_matrix[i, j] > np.nanmean(error_matrix) else 'black')

    ax2 = axes[1]

    im2 = ax2.imshow(bias_matrix, cmap='Oranges', aspect='auto', origin='lower')

    ax2.set_xlabel('Number of Hidden Layers', fontsize=12)

    ax2.set_ylabel('Number of Neurons per Layer', fontsize=12)

    ax2.set_title('Bias²', fontsize=14, fontweight='bold')

    ax2.set_xticks(range(len(layer_range)))

    ax2.set_xticklabels(layer_range)

    ax2.set_yticks(range(len(neuron_range)))

    ax2.set_yticklabels(neuron_range)

    cbar2 = plt.colorbar(im2, ax=ax2)

    cbar2.set_label('Bias²', fontsize=11)

    for i in range(len(neuron_range)):
        for j in range(len(layer_range)):
            if not np.isnan(bias_matrix[i, j]):
                text = ax2.text(j, i, f'{bias_matrix[i, j]:.4f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if bias_matrix[i, j] > np.nanmean(bias_matrix) else 'black')

    ax3 = axes[2]

    im3 = ax3.imshow(variance_matrix, cmap='Greens', aspect='auto', origin='lower')

    ax3.set_xlabel('Number of Hidden Layers', fontsize=12)

    ax3.set_ylabel('Number of Neurons per Layer', fontsize=12)

    ax3.set_title('Variance', fontsize=14, fontweight='bold')

    ax3.set_xticks(range(len(layer_range)))

    ax3.set_xticklabels(layer_range)

    ax3.set_yticks(range(len(neuron_range)))

    ax3.set_yticklabels(neuron_range)

    cbar3 = plt.colorbar(im3, ax=ax3)

    cbar3.set_label('Variance', fontsize=11)

    for i in range(len(neuron_range)):
        for j in range(len(layer_range)):
            if not np.isnan(variance_matrix[i, j]):
                text = ax3.text(j, i, f'{variance_matrix[i, j]:.4f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if variance_matrix[i, j] > np.nanmean(variance_matrix) else 'black')

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
