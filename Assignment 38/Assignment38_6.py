import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')

df['StudyHours'].hist()
plt.title("StudyHours Distribution")
plt.xlabel("StudyHours")
plt.ylabel("Frequency")
plt.show()
