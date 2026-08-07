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

for rs in [0,10,42]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    print(rs, accuracy_score(y_test, model.predict(X_test)))
