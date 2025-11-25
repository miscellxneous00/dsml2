# Use the covid_vaccine_statewise.csv dataset and perform the
# following analytics.
# A. Describe the dataset.
# B. Number of Males vaccinated
# C.. Number of females vaccinated

import pandas as pd

csv_path="./COVID-19 India Gender-wise.csv"

df=pd.read_csv(csv_path)

df.head()

df.describe()

df.info()

df.shape

df.columns

df.isnull().sum()

df.head()

print("Number of Males vaccinated")
df['Doses Administered - Male'].sum()

print("Number of Female vaccinated")
print(df['Doses Administered - Female'].sum())