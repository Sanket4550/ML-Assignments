import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_performance_ml.csv')

print("First 5 records:\n", df.head())
print("\nLast 5 records:\n", df.tail())
print("\nShape (rows, cols):", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
