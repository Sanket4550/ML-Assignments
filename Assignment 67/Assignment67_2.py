import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --------------------------------------------------
# 1. Create dataset
# --------------------------------------------------

X = np.array([
    [25000, 600, 200000, 10000, 0],
    [40000, 700, 300000, 8000, 1],
    [60000, 750, 500000, 12000, 1],
    [20000, 550, 150000, 15000, 0],
    [80000, 800, 700000, 10000, 1],
    [35000, 650, 250000, 9000, 1],
    [18000, 500, 100000, 12000, 0],
    [90000, 850, 800000, 15000, 1],
    [30000, 580, 200000, 14000, 1],
    [70000, 780, 600000, 10000, 1]
])

# Output:
# 0 = Loan rejected
# 1 = Loan approved

y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])

# --------------------------------------------------
# 2. Check / clean dataset
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
# 8. Evaluate model
# --------------------------------------------------

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test Accuracy:", accuracy * 100, "%")

# --------------------------------------------------
# 9. Predict new applicant
# --------------------------------------------------

new_applicant = np.array([
    [55000, 720, 400000, 10000, 1]
])

new_applicant_scaled = scaler.transform(new_applicant)

prediction = model.predict(new_applicant_scaled, verbose=0)

if prediction[0][0] >= 0.5:
    print("Prediction: Loan Approved")
else:
    print("Prediction: Loan Rejected")