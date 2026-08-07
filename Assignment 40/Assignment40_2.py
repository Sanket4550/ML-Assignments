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

df2 = df.drop('SleepHours', axis=1)
X2 = df2.drop('FinalResult', axis=1)
y2 = df2['FinalResult']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = DecisionTreeClassifier()
model2.fit(X_train2, y_train2)
print("Accuracy without SleepHours:", accuracy_score(y_test2, model2.predict(X_test2)))
