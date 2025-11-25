# Perform the following operations using Python on a suitable data set,
# counting unique values of data, format of each column, converting
# variable data type (e.g. from long to short, vice versa), identifying missing
# values and filling in the missing values.)

# Perform Data Cleaning, Data transformation using Python on any data
# set.


import pandas as pd
import numpy as np

data = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Alice", "Bob", "Charlie", "Bob", None, "Eve", "Frank", "Alice"],
    "age": [25, 34, None, 28, 40, 22, None, 30],
    "income": [55000.0, 72000.5, 30000.0, None, 120000.0, 48000.0, 60000.0, None],
    "signup_date": ["2024-01-15","2023-11-03","2024-06-10","2022-12-20",
                    None,"2024-07-01","2023-08-08","2024-01-15"],
    "is_subscribed": [True, False, True, False, True, None, False, True],
    "rating": [1000, 2000, 1500, 3000, 1200, 2500, 2000, 1000]
}

df=pd.DataFrame(data)

print(df)

df.info()

df=df.drop('id',axis=1)

print("===== ORIGINAL DATAFRAME =====")
df

print("\n===== UNIQUE VALUE COUNTS =====")
df.nunique(dropna=False)

print("\n===== COLUMN DATA TYPES (BEFORE) =====")
df.dtypes

df['signup_date'] = pd.to_datetime(df['signup_date'],errors='coerce')
df['rating'] = df['rating'].astype('int16')
df['is_subscribed'] = df['is_subscribed'].astype('boolean')

print("\n===== COLUMN DATA TYPES (AFTER CONVERSION) =====")
df.dtypes

print("\n===== MISSING VALUES (BEFORE FILLING) =====")
print(df.isnull().sum())

df

df['name'] = df['name'].fillna(df['name'].mode()[0])
df['age']=df['age'].fillna(df['age'].median())
df['income']=df['income'].fillna(df['income'].median())
df['signup_date']=df['signup_date'].fillna(df['signup_date'].min())
df['is_subscribed'] = df['is_subscribed'].fillna(False)

df

print("\n===== MISSING VALUES (After FILLING) =====")
print(df.isnull().sum())

df["income_per_age"] = df["income"] / df["age"]