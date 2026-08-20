import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# Dataset used for feature scaling
data = np.array([
    [25, 20000],
    [30, 40000],
    [35, 80000]
])

# Two points for which distance is calculated
point1 = data[0]
point2 = data[2]

# Euclidean distance before scaling
distance_before = euclidean(point1, point2)

# Apply StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_point1 = scaled_data[0]
scaled_point2 = scaled_data[2]

# Euclidean distance after scaling
distance_after = euclidean(scaled_point1, scaled_point2)

print("Point 1 before scaling:", point1)
print("Point 2 before scaling:", point2)
print("Euclidean distance before scaling:", distance_before)

print("\nPoint 1 after scaling:", scaled_point1)
print("Point 2 after scaling:", scaled_point2)
print("Euclidean distance after scaling:", distance_after)

print("\nExplanation:")
print("Before scaling, the second feature has much larger numerical values,")
print("so it dominates the Euclidean distance.")
print("After scaling, both features have comparable influence on the distance.")
