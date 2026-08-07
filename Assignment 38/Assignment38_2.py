import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')

print("Total students:", len(df))
print("Passed:", (df['FinalResult'] == 1).sum())
print("Failed:", (df['FinalResult'] == 0).sum())
