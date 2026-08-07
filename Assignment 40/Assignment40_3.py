import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('student_performance_ml.csv')

# Limited features
X_small = df[['StudyHours','Attendance']]
y = df['FinalResult']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

print("Accuracy with limited features:", acc)
print("Answer: Accuracy is usually lower because fewer features are used.")
