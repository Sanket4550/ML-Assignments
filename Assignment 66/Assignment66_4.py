# Input values
x = 2

# Initial weight and bias
weight = 0.5
bias = 0.1

# Target output
target = 2.0

# Learning rate
learning_rate = 0.1

# Calculate prediction
prediction = (x * weight) + bias

# Calculate error
error = target - prediction

# Store old weight
old_weight = weight

# Calculate gradient
gradient = error * x

# Update weight using gradient descent
weight = weight + (learning_rate * gradient)

# Display results
print("Prediction =", prediction)
print("Error =", error)
print("Old Weight =", old_weight)
print("Updated Weight =", weight)