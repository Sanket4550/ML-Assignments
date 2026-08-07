import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('student_performance_ml.csv')

X = df.drop('FinalResult', axis=1)
y = df['FinalResult']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("Feature Importance:\n")
for col, val in zip(X.columns, model.feature_importances_):
    print(f"{col} → {val:.4f}")

print("\nAnswer: Higher value = more important feature in prediction")
