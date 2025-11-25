import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path="./Titanic-Dataset.csv"

df=pd.read_csv(csv_path)

df.head()

df.info()

df.describe()

df.corr(numeric_only=True)['Survived']

plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Survived')
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Sex',hue='Survived')
plt.title("Survival Gender Wise")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df.dropna(),x='Pclass',hue='Survived')
plt.title("Passenger Class Wise")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data=df['Age'].dropna(),kde=True)
plt.title("Age Wise Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data=df['Fare'].dropna(),kde=True)
plt.title("Fare Wise Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df,x='Pclass',y='Age')
plt.title("Age vs Class")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['Survived','Age','Fare','Pclass']],hue='Survived')
plt.show()

#Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram.
plt.figure(figsize=(8,5))
plt.hist(df['Fare'].dropna(), bins=30, edgecolor='black')
plt.title("Distribution of Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.3)
plt.show()