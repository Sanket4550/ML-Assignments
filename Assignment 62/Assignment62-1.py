

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#--------------------------------------------
# 1. Load the dataset
#--------------------------------------------

df = pd.read_csv("Employee_Attrition.csv")

#--------------------------------------------
# 2. Display columns,shape and first five rows
#--------------------------------------------

print("Dataset shape : ")
print(df.shape)

print("\nDataset columns : ")
print(df.columns.tolist())

print("\nFirst 5 records : ")
print(df.head())

#--------------------------------------------
# 3. Cheack missing values
#--------------------------------------------

print("\nMissing values : ")
print(df.isnull().sum())

#--------------------------------------------
# 4. Identify numerical and categorical feature
#--------------------------------------------

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

categorical_features = [
    "OverTime",
    "Attrition"
]

print("\nnumerical features : ")
print(numerical_features)

print("\nCategorical features : ")
print(categorical_features)

#--------------------------------------------
# 5. Handle missing values
#--------------------------------------------

for column in numerical_features:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_features:
    df[column] = df[column].fillna(df[column].mode()[0])

#--------------------------------------------
# 6. Convert categorical into numeric
#--------------------------------------------

df[["OverTime","Attrition"]] = df[["OverTime","Attrition"]].map({
    "Yes" : 1,
    "No" : 0
})

#--------------------------------------------
# 7. Seperate Independant and dependant variables
#--------------------------------------------

features_columns = numerical_features + ["OverTime"]

X = df[features_columns]
Y = df["Attrition"]

#--------------------------------------------
# 8. Devide dataset into training and testing data
#--------------------------------------------

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

#--------------------------------------------
# 9. Apply feature scaling
#--------------------------------------------

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

#--------------------------------------------
# 10. Design MLP with two hidden layers
#--------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation="relu",
    random_state=42,
    max_iter=500,
    solver="adam"
)

#--------------------------------------------
# 11. Train the network
#--------------------------------------------

model.fit(X_train,Y_train)

#--------------------------------------------
# 12. Display no. of iteration
#--------------------------------------------

print("\nNo.of iteration required : ")
print(model.n_iter_)

#--------------------------------------------
# 13. Calculate training accuracy
#--------------------------------------------

Y_train_pred = model.predict(X_train)

training_accuracy = accuracy_score(Y_train,Y_train_pred)

print(f"Training accuracy : {training_accuracy * 100:.2f}%")

#--------------------------------------------
# 14. Calculate testing accuracy
#--------------------------------------------

Y_test_pred = model.predict(X_test)

testing_accuracy = accuracy_score(Y_test,Y_test_pred)

print(f"Testing accuracy : {testing_accuracy * 100:.2f}%")

#--------------------------------------------
# 15. generate confusion matrix
#--------------------------------------------

cm = confusion_matrix(Y_test,Y_test_pred)

print("\nConfusion matrix : ")
print(cm)


#--------------------------------------------
# 16. Plot loss curve
#--------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(model.loss_curve_)

plt.title("MLP training loss curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.grid()
plt.show()

#--------------------------------------------
# 17. Calculate testing accuracy
#--------------------------------------------

def PredictAttrition(employee_data):
    data = pd.DataFrame(
        [employee_data],
        columns=[
            "Age",
            "MonthlyIncome",
            "YearsAtCompany",
            "TotalWorkingYears",
            "DistanceFromHome",
            "JobSatisfaction",
            "WorkLifeBalance",
            "NumCompaniesWorked",
            "TrainingTimesLastYear",
            "OverTime"
        ]
    )

    #convert overtime
    data["OverTime"] = data["OverTime"].map({
        "Yes": 1,
        "No": 0
    })


    data_scaled = scalar.transform(data)

    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        return "1 -> Employee is likely to leave"
    else:
        return "0 -> Employee is likely to stay"

#--------------------------------------------
# 18. Test using 5 new employee data
#--------------------------------------------

new_employees = [
    [25,30000,1,2,10,2,2,"Yes",1,2],
    [40,70000,10,15,5,4,4,"No",2,4],
    [30,45000,3,5,20,2,2,"Yes",3,3],
    [35,60000,8,12,3,4,3,"No",1,3],
    [28,35000,2,4,15,1,2,"Yes",2,2]
]

print("\nPrediction for 5 new employees : ")

for i, employee in enumerate(new_employees,start=1):

    result = PredictAttrition(employee)

    print(f"Employee {i}: {result}")

#--------------------------------------------
# 19. Overfitting / underfitting analysis
#--------------------------------------------

print("\n Model Analysis")

accuracy_difference = training_accuracy - testing_accuracy

if accuracy_difference > 0.10:
    print("The model may be suffering from overfitting")

elif training_accuracy < 0.70 and testing_accuracy < 0.70:
    print("The model may be suffering from underfitting")

else:
    print("The model does not show overfitting and underfitting")

print("\n" + "-"*60)
print("Prediction completed successfully")
print("-"*60)
