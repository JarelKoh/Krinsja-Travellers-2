# Project 2 - Neural Networks and Regression

Group members: See Gek Cheryl Ong, Boon Kiat Khaw, Naden Jarel Anthony Koh

This project implements various machine learning techniques including neural networks, optimization algorithms, and regularization methods. The code is organized into modular Python files for each part of the project.

## File Structure in Code folder

```
.
├── main.py                    # Main script to run all parts
├── Project2_Part_b.py         # Part B: OLS, Adam, Neural Networks
├── Project2_Part_c.py         # Part C: Keras/TensorFlow implementation
├── Project2_Part_d.py         # Part D: Activation functions
├── Project2_Part_e.py         # Part E: Ridge and LASSO regression
├── Project2_Part_f.py         # Part F: MNIST classification
├── Project2_Notebook.ipynb    # Jupyter Notebook of Parts B-F
```

### Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed with the following packages:

```bash
pip install numpy matplotlib scikit-learn tensorflow autograd
```

### Running the Code

**Option 1: Run all parts in sequence (Recommended)**

```bash
python main.py
```

This will execute all parts B through F in order, showing progress and results for each part.

**Option 2: Run a specific part only**

```bash
python main.py B    # Run Part B only
python main.py C    # Run Part C only
python main.py D    # Run Part D only
python main.py E    # Run Part E only
python main.py F    # Run Part F only
```

**Option 3: Run individual part files directly**

```bash
python Project2_Part_b.py
python Project2_Part_c.py
python Project2_Part_d.py
python Project2_Part_e.py
python Project2_Part_f.py
```

## What Each Part Does

### Part B - Regression Fundamentals
- Ordinary Least Squares (OLS) polynomial regression
- Adam optimizer implementation for mini-batch training
- Neural network training on Runge function
- Comparison of different optimizers (SGD, Adam, RMSprop)

### Part C - Keras/TensorFlow Implementation
- Neural network using Keras/TensorFlow
- RMSprop optimizer with L2 regularization
- Hyperparameter tuning (learning rate, regularization)
- 3D visualization of predictions

### Part D - Activation Functions
- Testing different activation functions (sigmoid, ReLU, Leaky ReLU, tanh)
- Analysis of network depth and width effects
- Performance comparison across architectures

### Part E - Regularization Techniques
- Ridge regression implementation
- LASSO regression using ISTA algorithm
- Comparison with scikit-learn implementations
- Neural network training with L1/L2 regularization

### Part F - MNIST Classification
- Multi-class classification on MNIST dataset
- Softmax activation and cross-entropy loss
- Adam optimizer with bias correction
- Performance evaluation and visualization

## Important Notes

- **Part F (MNIST)**: The first run will automatically download the MNIST dataset (~11 MB). This may take a few minutes.
- **Computation Time**: Some parts (especially B, D, E, F) may take several minutes to complete due to training multiple models.
- **Visualizations**: All parts generate matplotlib plots. Close each plot window to continue execution.
- **Random Seeds**: Random seeds are set for reproducibility, so you should get consistent results across runs.

## Technical Details

Each `Project2_Part_X.py` file contains:
1. **Import statements** - All required libraries
2. **Function definitions** - Helper functions and model implementations
3. **main() function** - The execution code wrapped in a function
4. **if __name__ == "__main__":** - Allows standalone execution

This structure allows each part to be:
- Run independently as a script
- Imported as a module by `main.py`
- Easily tested and debugged

## Expected Output

When running `python main.py`, you should see:
- Progress messages for each part (B → C → D → E → F)
- Training metrics (loss, accuracy) during model training
- Matplotlib figures showing results and visualizations
- Summary statistics and final performance metrics
- Status indicators (✓ for success, ✗ for errors)

## Troubleshooting

**Issue: "Module not found" errors**
- Solution: Install missing packages using `pip install <package-name>`

**Issue: MNIST download fails**
- Solution: Check your internet connection and try again

**Issue: Out of memory errors**
- Solution: Reduce the training set size in the respective part file

**Issue: Plots don't show**
- Solution: Make sure you have a display/GUI environment available

