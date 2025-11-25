# Use the covid_vaccine_statewise.csv dataset and perform the
# following analytics.
# a. Describe the dataset
# b. Number of persons state wise vaccinated for first dose in India
# c. Number of persons state wise vaccinated for second dose in India

import pandas as pd

csv_path="./COVID-19 India Statewise Vaccine Data.csv"

df=pd.read_csv(csv_path)

df

df.columns

df.info()

df.describe()

df.shape

df.head()

print("Number of persons state wise vaccinated for first dose in India")
df.groupby("State/UTs")['Dose1'].sum().sort_values(ascending=False)

print("Number of persons state wise vaccinated for second dose in India")
print(df.groupby('State/UTs')['Dose 2'].sum().sort_values(ascending=False))