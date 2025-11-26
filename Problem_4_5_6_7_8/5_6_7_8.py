#Write a program to do: A dataset collected in a cosmetics shop showing
# details of customers and whether or not they responded to a special offer
# to buy a new lip-stick is shown in table below. (Use library commands)
# According to the decision tree you have made from the previous training
# data set, what is the decision for the test data: [Age < 21, Income = Low,
# Gender = Female, Marital Status = Married]?

# python3 -m pip install scikit-learn

import pandas as pd
import matplotlib.pyplot as plt

csv_path="./Lipstick.csv"

df=pd.read_csv(csv_path)

df.head()

df['Age']

# df['Age']=df['Age'].replace({"<21":"Young","21-35":"Adult",">35":"Old"})

df=df.drop('Id',axis=1)

df.info()

df.describe()

from sklearn.model_selection import train_test_split

X=pd.get_dummies(df.drop('Buys',axis=1),drop_first=True)
y=df['Buys']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.tree import DecisionTreeClassifier,plot_tree

model = DecisionTreeClassifier()

model.fit(X,y)

plt.figure(figsize=(10,8),dpi=200)
plot_tree(model,feature_names=X.columns,filled=True)

from sklearn.metrics import classification_report,confusion_matrix

y_pred=model.predict(X_test)

print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred)

test_1 = pd.DataFrame([{
    "Age": "<21",
    "Income": "Low",
    "Gender": "Female",
    "Marital Status": "Married"
}])

sample_X = pd.get_dummies(test_1)
for col in X.columns:
    if col not in sample_X.columns:
        sample_X[col] = 0
sample_X = sample_X[X.columns]   # Arrange in order

print(model.predict(sample_X))

test_2 = pd.DataFrame([{
    "Age": ">35",
    "Income": "Medium",
    "Gender": "Female",
    "Marital Status": "Married"
}])

sample_X = pd.get_dummies(test_2)
for col in X.columns:
    if col not in sample_X.columns:
        sample_X[col] = 0
sample_X = sample_X[X.columns]   # Arrange in order

print(model.predict(sample_X))

test_3 = pd.DataFrame([{
    "Age": "21-35",
    "Income": "Low",
    "Gender": "Male",
    "Marital Status": "Married"
}])

sample_X = pd.get_dummies(test_3)
for col in X.columns:
    if col not in sample_X.columns:
        sample_X[col] = 0
sample_X = sample_X[X.columns]   # Arrange in order

print(model.predict(sample_X))