import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')

counts = df['FinalResult'].value_counts()
percent = df['FinalResult'].value_counts(normalize=True) * 100
print("Counts:\n", counts)
print("\nPercentage:\n", percent)
