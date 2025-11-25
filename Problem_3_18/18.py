#Use House_Price prediction dataset. Provide summary statistics (mean, median, minimum, maximum, standard deviation) of variables (categorical vs quantitative) such as- For example, if categorical variable is age groups and quantitative variable is income, then provide summary statistics of income grouped by the age groups.

import pandas as pd

csv_path="./House Price Prediction Dataset.csv"

df=pd.read_csv(csv_path)

df.head()

df.info()

df.describe()

df.drop('Id',axis=1).mean(numeric_only=True)

df.drop('Id',axis=1).std(numeric_only=True)

df.drop('Id',axis=1).var(numeric_only=True)

df.drop('Id',axis=1).min(numeric_only=True)

df.drop('Id',axis=1).max(numeric_only=True)

df.select_dtypes(include=['object'])

numeric_columns=df.select_dtypes(exclude=['object']).columns.to_list()
numeric_columns.remove('Id')

categorical_columns=df.select_dtypes(include=['object']).columns

for category in categorical_columns:
    for numeric in numeric_columns:
        print(f"\n=== {numeric} grouped by {category} ===\n")
        print(df.groupby(category)[numeric].agg(['mean','median','min','max','std','var']))