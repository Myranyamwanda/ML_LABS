import numpy as np

# 1. Generate 200 samples of x and corresponding samples of y using NumPy
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(200, 1)
# True slope = 3, True bias/intercept = 4, plus some random noise
y = 4 + 3 * X + np.random.randn(200, 1)

# Hyperparameters
learning_rate = 0.1
iterations = 1000
m = len(X)

# Initialize coefficients randomly
slope = np.random.randn(1)[0]
bias = np.random.randn(1)[0]

# 2. Arrays to store history for MSE, Slope, and Bias values
mse_history = np.zeros(iterations)
slope_history = np.zeros(iterations)
bias_history = np.zeros(iterations)

# Gradient Descent Technique
for i in range(iterations):
    # Calculations / Predictions
    y_pred = (slope * X) + bias
    
    # Calculate Mean Squared Error (Cost)
    mse = (1 / m) * np.sum((y_pred - y) ** 2)
    
    # Compute Gradients
    d_slope = (2 / m) * np.sum((y_pred - y) * X)
    d_bias = (2 / m) * np.sum(y_pred - y)
    
    # Update coefficients using the learning rate
    slope = slope - (learning_rate * d_slope)
    bias = bias - (learning_rate * d_bias)
    
    # Store values in arrays
    mse_history[i] = mse
    slope_history[i] = slope
    bias_history[i] = bias

# 3. Return the optimal values of Slope and Bias that have the lowest cost (MSE)
lowest_cost_index = np.argmin(mse_history)
optimal_slope = slope_history[lowest_cost_index]
optimal_bias = bias_history[lowest_cost_index]

# Print out the optimal values to the Terminal
print("      GRADIENT DESCENT OPTIMAL VALUES    ")
print(f"Optimal Slope: {optimal_slope:.4f}")
print(f"Optimal Bias:  {optimal_bias:.4f}")
print(f"Lowest MSE:    {mse_history[lowest_cost_index]:.4f}")
