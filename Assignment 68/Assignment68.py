

# Q1. Manual Convolution

image = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

kernel = [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]

feature_map = []

for i in range(len(image) - 2):
    row = []
    for j in range(len(image[0]) - 2):
        total = 0

        for x in range(3):
            for y in range(3):
                total = total + image[i + x][j + y] * kernel[x][y]

        row.append(total)

    feature_map.append(row)

print("Feature Map:")
for row in feature_map:
    print(row)


# Q2. ReLU and Max Pooling

feature_map = [
    [3, 3, 3],
    [0, 0, 0],
    [-3, -3, -3]
]

relu_output = []

for row in feature_map:
    new_row = []

    for value in row:
        if value < 0:
            new_row.append(0)
        else:
            new_row.append(value)

    relu_output.append(new_row)

print("\nReLU Output:")
for row in relu_output:
    print(row)

pool_size = 2
pooled_output = []

for i in range(0, len(relu_output) - 1, pool_size):
    row = []

    for j in range(0, len(relu_output[0]) - 1, pool_size):
        maximum = relu_output[i][j]

        for x in range(pool_size):
            for y in range(pool_size):
                if relu_output[i + x][j + y] > maximum:
                    maximum = relu_output[i + x][j + y]

        row.append(maximum)

    pooled_output.append(row)

print("\nMax Pooling Output:")
for row in pooled_output:
    print(row)

print("\nMax Pooling reduces the size of the feature map while keeping the maximum important value.")


# Q3. Flattening

matrix = [
    [6, 4],
    [8, 6]
]

flatten_output = []

for row in matrix:
    for value in row:
        flatten_output.append(value)

print("\nFlatten Output:")
print(flatten_output)

print("\nFlattening converts a 2D feature map into a 1D vector.")
print("This vector can be given as input to a fully connected layer.")
