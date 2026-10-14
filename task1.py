import pandas as pd
df = pd.read_csv("Titanic.csv")
#print(df.isnull().sum())
df=df.drop_duplicates()
print(df.duplicated().sum())
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.dtypes)
df.to_csv("Cleaned_Titanic.csv", index=False)