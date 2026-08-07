import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')
X = df.drop('FinalResult', axis=1)
y = df['FinalResult']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

new_students = pd.DataFrame({
    'StudyHours':[5,6,7,4,8],
    'Attendance':[80,85,90,70,95],
    'PreviousScore':[60,65,70,55,75],
    'AssignmentsCompleted':[6,7,8,5,9],
    'SleepHours':[6,7,8,5,7]
})
pred = model.predict(new_students)
print(pred)
