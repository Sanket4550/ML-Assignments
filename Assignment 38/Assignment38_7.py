import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')

colors = df['FinalResult'].map({1: 'green', 0: 'red'})
plt.scatter(df['StudyHours'], df['PreviousScore'], c=colors)
plt.xlabel("StudyHours")
plt.ylabel("PreviousScore")
plt.title("StudyHours vs PreviousScore")
plt.show()
