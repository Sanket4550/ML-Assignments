import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --------------------------------------------------
# 1. Create dataset
# Features:
# [Age, Monthly Charges, Tenure, Complaints, Support Calls]
# --------------------------------------------------

X = np.array([
    [25, 500, 12, 1, 2],
    [30, 700, 24, 0, 1],
    [45, 1200, 6, 5, 8],
    [50, 1500, 5, 6, 10],
    [28, 600, 18, 1, 1],
    [35, 800, 30, 0, 0],
    [48, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 550, 20, 0, 1],
    [42, 1300, 8, 4, 7]
])

# Output:
# 0 = Customer will stay
# 1 = Customer will leave

y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])

# --------------------------------------------------
# 2. Clean the dataset
# --------------------------------------------------

print("Missing values:", np.isnan(X).sum())

# --------------------------------------------------
# 3. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Apply StandardScaler
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 5. Create FNN model
# --------------------------------------------------

model = Sequential([
    Dense(16, activation='relu', input_shape=(5,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# --------------------------------------------------
# 6. Compile model
# --------------------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=2,
    verbose=0
)

# --------------------------------------------------
# 8. Evaluate accuracy
# --------------------------------------------------

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Accuracy:", accuracy * 100, "%")

# --------------------------------------------------
# 9. Test new customer
# --------------------------------------------------

new_customer = np.array([[46, 1450, 5, 6, 9]])

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled, verbose=0)

if prediction[0][0] >= 0.5:
    print("Prediction: Customer will leave")
else:
    print("Prediction: Customer will stay")