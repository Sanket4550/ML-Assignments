import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('student_performance_ml.csv')
X = df.drop('FinalResult', axis=1)
y = df['FinalResult']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

sample = [[6, 85, 66, 7, 7]]  # StudyHours, Attendance, PreviousScore, AssignmentsCompleted, SleepHours
prediction = model.predict(sample)
print("Prediction (1=Pass, 0=Fail):", prediction[0])
