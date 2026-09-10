import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#_______________________________________________________
#
# 1. Load dataset
#________________________________________________________

df = pd.read_csv("loan_default.csv")

print(df.head())
print(df.info())

#__________________________________________________________
#
# 2. Check missing 
#__________________________________________________________

print("\nMissing Values:")
print(df.isnull().sum())

#_________________________________________________________
#
# 3. Remove missing rows
#_________________________________________________________
df = df.dropna()

#_________________________________________________________
3
# 4. Encode categorical columns
#__________________________________________________________
le = LabelEncoder()

df["PreviousDefault"] = le.fit_transform(df["PreviousDefault"])
df["HomeOwnership"] = le.fit_transform(df["HomeOwnership"])

#___________________________________________________________
#
# 5. Separate X and y
#____________________________________________________________
X = df.drop("Default", axis=1)
y = df["Default"]


# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 8. Create MLP model
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)


# 9. Train model
model.fit(X_train, y_train)


# 10. Prediction
y_pred = model.predict(X_test)


# 11. Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))


# 12. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 13. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 14. Test a new applicant
new_customer = [[
    35,       # Age
    600000,   # Income
    250000,   # LoanAmount
    720,      # CreditScore
    8,        # EmploymentYears
    1,        # ExistingLoans
    15000,    # MonthlyDebt
    60,       # LoanTerm
    0,        # PreviousDefault
    1         # HomeOwnership
]]

new_customer = scaler.transform(new_customer)

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nHigh Default Risk")
else:
    print("\nLow Default Risk")