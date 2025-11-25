import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path="./House Price Prediction Dataset.csv"

df=pd.read_csv(csv_path)

df.head()

df.info()

numeric_cols = df.select_dtypes(exclude=['object']).columns
numeric_cols

print("=== STANDARD DEVIATION FOR EACH NUMERIC FEATURE ===")
df[numeric_cols].std()

print("=== Variance FOR EACH NUMERIC FEATURE ===")
df[numeric_cols].var()

print("=== Percentiles FOR EACH NUMERIC FEATURE ===")
df[numeric_cols].quantile([0.25,0.5,0.75])

print("Plotting Histograms...")
df[numeric_cols].hist(figsize=(15,12),bins=20)
plt.tight_layout()
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col].dropna(), bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()