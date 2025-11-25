import pandas as pd

csv_path="./churn-bigml-80.csv"

df=pd.read_csv(csv_path)

df.head()

X=df.drop('Churn',axis=1)

print("=== Minimum Value of Each Column ===")
X.min(numeric_only=True)

print("=== Maximum Value of Each Column ===")
X.max(numeric_only=True)

print("=== Mean Value of Each Column ===")
X.mean(numeric_only=True)

print("=== Range of Each Column ===")
X.max(numeric_only=True) - df.min(numeric_only=True)

print("=== Standard deviation of Each Column ===")
X.std(numeric_only=True)

print("=== Variance of Each Column ===")
X.var(numeric_only=True)

print("=== Percentiles (25th, 50th, 75th) ===")
X.quantile([0.25,0.5,0.75],numeric_only=True)