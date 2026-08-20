import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([
    [25, 20000],
    [30, 40000],
    [35, 80000]
])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("Original Dataset:")
print(data)

print("\nScaled Dataset:")
print(scaled_data)
