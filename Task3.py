import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Cleaned_Titanic.csv")
df["Pclass"].value_counts().plot(kind="bar")

plt.title("Passengers by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

plt.hist(df["Age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

df["Survived"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Survival Distribution")
plt.ylabel("")
plt.show()

plt.scatter(df["Age"], df["Fare"])

plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

df["Fare"].head(20).plot(kind="line")

plt.title("Fare of First 20 Passengers")
plt.xlabel("Passenger")
plt.ylabel("Fare")
plt.show()

sns.countplot(data=df, x="Sex", hue="Survived")

plt.title("Survival by Gender")
plt.show()