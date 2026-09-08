# Employee_Attrition.py
# Deep Learning-Based Employee Attrition Prediction System
# Using MLPClassifier

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------------------------------------------------
# 1. Load the dataset using Pandas
# ---------------------------------------------------------
df = pd.read_csv("Employee_Attrition.csv")

print("=" * 60)
print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("=" * 60)

print("\n1. Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Records:")
print(df.head())


# ---------------------------------------------------------
# 2. Check for missing values
# ---------------------------------------------------------
print("\n2. Missing Values:")
print(df.isnull().sum())

# Handle missing values if any
numerical_features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "TotalWorkingYears",
    "DistanceFromHome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "NumCompaniesWorked",
    "TrainingTimesLastYear"
]

categorical_features = ["OverTime"]

for column in numerical_features:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_features:
    df[column] = df[column].fillna(df[column].mode()[0])


# ---------------------------------------------------------
# 3. Identify numerical and categorical features
# ---------------------------------------------------------
print("\n3. Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ---------------------------------------------------------
# 4. Convert categorical feature OverTime into 0/1
# ---------------------------------------------------------
df["OverTime"] = df["OverTime"].map({
    "Yes": 1,
    "No": 0
})


# ---------------------------------------------------------
# 5. Convert target Attrition into 0/1
# ---------------------------------------------------------
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# ---------------------------------------------------------
# 6. Separate independent and dependent variables
# ---------------------------------------------------------
feature_columns = numerical_features + ["OverTime"]

X = df[feature_columns]
y = df["Attrition"]


# ---------------------------------------------------------
# 7. Divide dataset into training and testing data
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 8. Apply feature scaling
# ---------------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------
# 9. Design MLP with at least two hidden layers
# ---------------------------------------------------------
# Hidden Layer 1 = 8 neurons
# Hidden Layer 2 = 4 neurons

model = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)


# ---------------------------------------------------------
# 10. Train the network
# ---------------------------------------------------------
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 11. Display number of iterations required for training
# ---------------------------------------------------------
print("\n4. Number of Iterations Required:")
print(model.n_iter_)


# ---------------------------------------------------------
# 12. Calculate training accuracy
# ---------------------------------------------------------
y_train_pred = model.predict(X_train)

training_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\n5. Training Accuracy:")
print(f"{training_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# 13. Calculate testing accuracy
# ---------------------------------------------------------
y_test_pred = model.predict(X_test)

testing_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\n6. Testing Accuracy:")
print(f"{testing_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# 14. Generate confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)

print("\n7. Confusion Matrix:")
print(cm)

print("\nConfusion Matrix Format:")
print("             Predicted")
print("             Stay  Leave")
print(f"Actual Stay  {cm[0][0]:4d}  {cm[0][1]:5d}")
print(f"Actual Leave {cm[1][0]:4d}  {cm[1][1]:5d}")


# ---------------------------------------------------------
# 15. Plot the loss curve
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()
plt.show()


# ---------------------------------------------------------
# 16. Function to predict employee attrition
# ---------------------------------------------------------
def PredictAttrition(employee_data):
    """
    employee_data must contain values in this order:

    Age
    MonthlyIncome
    YearsAtCompany
    TotalWorkingYears
    DistanceFromHome
    JobSatisfaction
    WorkLifeBalance
    OverTime
    NumCompaniesWorked
    TrainingTimesLastYear
    """

    data = pd.DataFrame(
        [employee_data],
        columns=feature_columns[:-1] + ["OverTime"]
    )

    # Convert OverTime to numerical representation
    data["OverTime"] = data["OverTime"].map({
        "Yes": 1,
        "No": 0
    })

    # Scale the new employee data
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        return "1 -> Employee is likely to leave"
    else:
        return "0 -> Employee is likely to stay"


# ---------------------------------------------------------
# 17. Test the system using five new employee records
# ---------------------------------------------------------
new_employees = [
    [25, 30000, 1, 2, 10, 2, 2, "Yes", 1, 2],
    [40, 70000, 10, 15, 5, 4, 4, "No", 2, 4],
    [30, 45000, 3, 5, 20, 2, 2, "Yes", 3, 3],
    [35, 60000, 8, 12, 3, 4, 3, "No", 1, 3],
    [28, 35000, 2, 4, 15, 1, 2, "Yes", 2, 2]
]

print("\n8. Predictions for Five New Employees:")

for i, employee in enumerate(new_employees, start=1):
    result = PredictAttrition(employee)
    print(f"Employee {i}: {result}")


# ---------------------------------------------------------
# 18. Explain overfitting or underfitting
# ---------------------------------------------------------
print("\n9. Model Analysis:")

accuracy_difference = training_accuracy - testing_accuracy

if accuracy_difference > 0.10:
    print("The model may be suffering from OVERFITTING.")
    print("Training accuracy is considerably higher than testing accuracy.")

elif training_accuracy < 0.70 and testing_accuracy < 0.70:
    print("The model may be suffering from UNDERFITTING.")
    print("Both training and testing accuracies are low.")

else:
    print("The model does not show significant overfitting or underfitting.")


print("\n" + "=" * 60)
print("Prediction completed successfully.")
print("=" * 60)
