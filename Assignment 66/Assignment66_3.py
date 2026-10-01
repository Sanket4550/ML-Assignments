import math

# Actual and predicted values
actual = [1, 0, 1, 1]
predicted = [0.9, 0.2, 0.8, 0.7]

# ---------------- MSE ----------------

mse = 0

for y, p in zip(actual, predicted):
    mse += (y - p) ** 2

mse = mse / len(actual)

print("Mean Squared Error =", mse)


# ---------------- Binary Cross Entropy ----------------

bce = 0

for y, p in zip(actual, predicted):
    bce += -(y * math.log(p) + (1 - y) * math.log(1 - p))

bce = bce / len(actual)

print("Binary Cross Entropy =", bce)