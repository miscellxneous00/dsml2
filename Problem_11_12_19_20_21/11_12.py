# Use Iris flower dataset and perform following :
# 1. List down the features and their types (e.g., numeric, nominal)
# available in the dataset. 2. Create a histogram for each feature in the
# dataset to illustrate the feature distributions.

# Use Iris flower dataset and perform following :
# 1. Create a box plot for each feature in the dataset.
# 2. Identify and discuss distributions and identify outliers from them.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "./IRIS 2.csv"

df=pd.read_csv(csv_path)

df.info()

df.describe()

df.head()

numeric_cols=df.select_dtypes(include=['int64','float64']).columns

numeric_cols

categorical_cols=df.select_dtypes(include=['object']).columns

categorical_cols

print("Numeric Features:")
for col in numeric_cols:
    print(col)

print("Categorical (Nominal) Features:")
for col in categorical_cols:
    print(col)

plt.figure(figsize=(6,4))
plt.hist(df['sepal_length'].dropna(),edgecolor="black")
plt.title(f"Histogram of SepalLengthCm")
plt.xlabel('sepal_length')
plt.ylabel('frequency')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col].dropna(),edgecolor="black")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col].dropna())
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()