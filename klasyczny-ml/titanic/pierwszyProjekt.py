import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
pd.set_option('display.max_columns', None)
print(df.sample(5))

print(f"Mediana wieku: ",  df["Age"].median())
print(f"Średnia wieku: ",  round(df["Age"].mean(), 3))
print("Mediana wieku ze wględu na płeć: ")
print(df.groupby("Sex")["Age"].median())
print("Mediana wieku ze względu na Pclass")
print(df.groupby("Pclass")["Age"].median())

print("Mediana ze względu na płeć i klasę: ")
medianOfAgeBySexPclass = df.groupby(["Sex", "Pclass"]).agg(
    median=("Age", "median")
)

print(medianOfAgeBySexPclass)
df = df.drop(columns="Cabin")
df = df.dropna(subset=["Embarked"])

median_age = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
print(median_age)

df["Age"] = df["Age"].fillna(median_age)

print(df.isna().sum())
print(df.shape)
print()

#zad. 1
survived = df["Survived"].sum()
died = df.loc[df["Survived"] == 0, "Survived"].count()
print("Przeżyło: ", survived)
print("Nie przeżyło: ", died)
print(f"Procent pasażerów jaki przezył: {round(survived/(survived+died)*100, 2)}%")

#zad. 2
survivabilityBySex = df.groupby("Sex").agg(
    liczba_osob=("Age", "count"),
    przezyli=("Survived", "sum"),
)

survivabilityBySex["SurvivalRate"] = round((survivabilityBySex["przezyli"] / survivabilityBySex["liczba_osob"])*100, 2)

print(survivabilityBySex)

#zad. 3
survivabilityByPclass = df.groupby("Pclass").agg(
    liczba_osob=("Age", "count"),
    przezyli=("Survived", "sum")
)

survivabilityByPclass["SurvivalRate"] = round(df.groupby("Pclass")["Survived"].mean(), 2)*100

print(survivabilityByPclass)

#zad. 4
survivabilityBySexPclass = df.groupby(["Sex", "Pclass"]).agg(
    ilosc_osob=("Age", "count"),
    przezyli=("Survived", "sum")
)

survivabilityBySexPclass["SurvivalRate"] = round(survivabilityBySexPclass["przezyli"] / survivabilityBySexPclass["ilosc_osob"] * 100, 2)

print(survivabilityBySexPclass)

#zad. 5
print("Średni wiek osób, które przeżyły: ", round(df.loc[df["Survived"] == 1, "Age"].mean(), 2))
print("Mediana wieku osób, które przeżyły", round(df.loc[df["Survived"] == 1, "Age"].median(), 2))
print("Średni wiek osób, które nie przeżyły: ", round(df.loc[df["Survived"] == 0, "Age"].mean(), 2))
print("Mediana wieku osób, które nie przeżyły", round(df.loc[df["Survived"] == 0, "Age"].median(), 2))

#MATPLOTLIB

"""
#zad. 1
#plt.hist(df["Age"], bins=30)
plt.title("Histogram wieku")
plt.xlabel("Wiek")
plt.ylabel("Liczba osób")
#plt.show()

#zad. 2
#plt.hist([df.loc[df["Survived"] == 1, "Age"], df.loc[df["Survived"] == 0, "Age"]], label=["Przeżyli", "Nie przeżyli"], alpha=0.5, bins=7)
plt.xlabel("Wiek")
plt.ylabel("Liczba osób")
#plt.show()

"""

labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"]
df["Range"] = pd.cut(
    df["Age"],
    bins=[0, 10, 20, 30, 40, 50, 60, 100],
    labels=labels)

survivabilityByAge = df.groupby("Range").agg(
    liczbaOsob=("Age", "count"),
    liczbaOcalalych=("Survived", "sum")
)

survivabilityByAge["SurvivalRateByAge"] = round((survivabilityByAge["liczbaOcalalych"] / survivabilityByAge["liczbaOsob"] * 100), 2)

print(survivabilityByAge)

survivabilityBySexRange = df.groupby(["Sex", "Range"]).agg(
    liczbaOsob=("Age", "count"),
    liczbaOcalalych=("Survived", "sum"),
)

survivabilityBySexRange["SurvivalRate"] = round((survivabilityBySexRange["liczbaOcalalych"] / survivabilityBySexRange["liczbaOsob"] * 100), 2)

print(survivabilityBySexRange)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["IsAlone"] = df["FamilySize"] == 1

survivabilityByIsAlone = df.groupby("IsAlone").agg(
    liczbaOsob=("Age", "count"),
    liczbaOcalalych=("Survived", "sum"),
    survivalRate=("Survived", "mean")
)

print(survivabilityByIsAlone)

survivabilityByFamilySize = df.groupby("FamilySize").agg(
    liczbaOsob=("Age", "count"),
    liczbaOcalalych=("Survived", "sum"),
)

survivabilityByFamilySize["SurvivalRate"] = round((survivabilityByFamilySize["liczbaOcalalych"] / survivabilityByFamilySize["liczbaOsob"] * 100), 2)

print(survivabilityByFamilySize)

"""
plt.bar(
    survivabilityByFamilySize.index,
    survivabilityByFamilySize["SurvivalRate"],
)
plt.xlabel("Liczba członków rodziny")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate w zależności od wielkości rodziny")
plt.show()

"""

survivabilityByFare = df.groupby("Survived")["Fare"].agg(["mean", "median", "min", "max"])
print(survivabilityByFare)

print(df.groupby("Pclass")["Fare"].agg(["mean", "median", "min", "max"]))

"""
plt.bar(survivabilityBySex.index, survivabilityBySex["SurvivalRate"])
plt.ylim(0, 100)
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate and Sex")
plt.show()
"""

"""
plt.bar(survivabilityByPclass.index, survivabilityByPclass["SurvivalRate"])
plt.ylim(0, 100)
plt.xticks([1, 2, 3])
plt.xlabel("Pclass")
plt.ylabel("Survival rate (%)")
plt.title("Pclass and Survival rate")
plt.show()
"""

import numpy as np


x = np.arange(3)
width = 0.35
female = survivabilityBySexPclass.loc["female", "SurvivalRate"]
male = survivabilityBySexPclass.loc["male", "SurvivalRate"]

"""
plt.xticks(x, [1, 2, 3])
plt.bar(x - width/2, female, width, label="Female")
plt.bar(x + width/2, male, width, label="Male")
plt.legend()
plt.xlabel("Sex and Pclass")
plt.ylabel("Survival Rate (%)")
plt.title("Survival Rate by Sex and Pclass")
plt.show()
"""

"""
plt.boxplot([
    df.loc[df["Survived"] == 1, "Age"],
    df.loc[df["Survived"] == 0, "Age"]
])
plt.xticks([1, 2], ["Przeżyli", "Nie przeżyli"])
plt.ylabel("Wiek")
plt.title("Rozkład wieku a przeżywalność")
plt.show()
"""

"""
plt.boxplot([
    df.loc[df["Survived"] == 1, "Fare"],
    df.loc[df["Survived"] == 0, "Fare"],
])
plt.xticks([1, 2], ["Przeżyli", "Nie przeżyli"])
plt.ylabel("Fare")
plt.show()
"""

"""
plt.hist(df["Fare"], bins=100)
plt.ylabel("Ilośc kupionych biletów")
plt.xlabel("Cena biletu")
plt.title("Histogram dla cen biletów")
plt.show()
"""

print(df.groupby("Pclass")["Name"].count())

"""
#sns.countplot(data2=df, x="Pclass")
sns.countplot(
    data2=df,
    x="Pclass",
    hue="Sex"
)

sns.barplot(
    data2=df,
    x="Pclass",
    y="Survived",
    hue="Sex"
)

sns.histplot(
    data2=df,
    x="Age",
    hue="Sex",
    bins=10,
    alpha=0.5
)

sns.scatterplot(
    data2=df,
    x="Age",
    y="Fare",
    hue="Pclass"
)

correlation = df[[
    "Survived",
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize"
]].corr()

sns.heatmap(
    data2=correlation,
    annot=True,
    cmap="coolwarm"
)
sns.pairplot(
    df,
    vars=["Age", "Fare", "Pclass"],
    hue="Survived"
)
plt.show()
"""