# Write a Python program to display some basic statistical details like
# percentile, mean, standard deviation etc (Use python and pandas
# commands) the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
# of iris.csv dataset.

import pandas as pd

csv_path = "./Iris.csv"

df=pd.read_csv(csv_path)

df.head()

df['Species'].value_counts()

species = ['Iris-setosa','Iris-versicolor','Iris-virginica']

for sp in species:
    print(f"\n========= Statistics for {sp} =========\n")

    species_df = df[df["Species"] == sp]
    species_df = species_df.drop("Id", axis=1)

    print("Summary Statistics:\n", species_df.describe(), "\n")

    print("Mean:\n",species_df.mean(numeric_only=True), "\n")

    print("Standard Deviation:\n", species_df.std(numeric_only=True), "\n")

    print("25th, 50th, 75th Percentiles:\n",
          species_df.quantile([0.25, 0.50, 0.75], numeric_only=True), "\n")
