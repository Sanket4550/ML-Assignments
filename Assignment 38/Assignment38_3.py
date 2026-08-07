import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')

print("Average StudyHours:", df['StudyHours'].mean())
print("Average Attendance:", df['Attendance'].mean())
print("Max PreviousScore:", df['PreviousScore'].max())
print("Min SleepHours:", df['SleepHours'].min())
