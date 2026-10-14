import pandas as pd

df = pd.read_csv("Cleaned_Titanic.csv")
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe(include="all"))
print(df["Sex"].value_counts())
print(df["Pclass"].value_counts())
print(df["Embarked"].value_counts())
print(df["Survived"].value_counts())
print(df["Survived"].value_counts(normalize=True) * 100)
print(df["Age"].describe())