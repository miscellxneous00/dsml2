#python --version
#pip install pandas
# python3 -m pip install pandas

#Perform the following operations using Python on a data set : read data from different formats(like csv, xls),indexing and selecting data, sort data, describe attributes of data, checking data types of each column. (Use Titanic Dataset)

# 1. Import required libraries
import pandas as pd


# 2. Read data from different formats
csv_path="./Titanic-Dataset.csv"


#Read CSV (Titanic dataset)
df=pd.read_csv(csv_path)


# 3. Display first few rows of the dataset
df.head(5)


# 4. Indexing and Selecting Data

# number of rows in dataset
print(f"Number of Rows : {df.shape[0]}")

# number of rows in dataset
print(f"Number of Columns : {df.shape[1]}")

# Select a single column
print("Selecting column 'Survived':")
print(df["Survived"].head(), "\n")

# Select multiple columns
df[['Name','Survived','Sex']].head(5)

# Select rows using iloc (index-based)
print("Row at index 3 (iloc):")
df.iloc[3]

# Select rows using loc (label-based)
print("Rows 20–30 using loc:")
df.iloc[20:30]

# select data using multiple conditions
df[(df['Survived']==1) & (df['Sex']=='female')]


# 5. Sorting Data
print("Sorting by Age (ascending):")
df.sort_values('Age')

print("Sorting by Fare (descending):")
df.sort_values('Fare')

# 6. Describing Attributes

print("Statistical Summary of Numerical Columns:")
df.describe()

print("Summary of Categorical Columns:")
print(df.describe(include='object'))

# Counting number male and female entries
df['Sex'].value_counts()

# Counting number Passenger class entries
df['Pclass'].value_counts()

#Basic overall info
df.info()

#checking number of null values in data set
df.isnull().sum()

print("Data Types of Columns:")
print(df.dtypes)