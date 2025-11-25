# With reference to Table , obtain the Frequency table for the
# attribute age. From the frequency table you have obtained, calculate
# the information gain of the frequency table while splitting on Age. (Use
# step by step Python/Pandas commands)

import pandas as pd
df=pd.read_csv('./dataset.csv')

print(df)

freq_age = df["Age"].value_counts()
print(freq_age)

import numpy as np

def entropy(column):
    counts = column.value_counts()
    probabilities = counts / len(column)
    return -np.sum(probabilities * np.log2(probabilities))

entropy_total = entropy(df["Class"])
print(entropy_total)

entropy_age = df.groupby("Age")["Class"].apply(entropy)
print(entropy_age)


total = len(df)

weighted_entropy = sum([
    (df[df["Age"] == age].shape[0] / total) * entropy_val
    for age, entropy_val in entropy_age.items()
])

print(weighted_entropy)


information_gain_age = entropy_total - weighted_entropy
print("Result:")
print(information_gain_age)
