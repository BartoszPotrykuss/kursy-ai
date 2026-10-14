import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name": [
        "Anna", "Jan", "Ola", "Piotr",
        "Kasia", "Jan", "Marek", "Ewa"
    ],

    "Age": [
        20, np.nan, 19, -5,
        22, np.nan, 150, 25
    ],

    "City": [
        "Kraków", "Gdańsk", None, "Poznań",
        "Gdańsk", "Gdańsk", "Warszawa", "Poznań"
    ],

    "Salary": [
        5000,
        7000,
        np.nan,
        6500,
        -1000,
        7000,
        12000,
        np.nan
    ]
})

#zad. 1
print(df.isna().sum())
print(df.isna().sum().sum())
print()

#zad. 2
print(df[df.isna().any(axis=1)])
print()

#zad.3
print(df[df["Age"] < 0])
print(df[df["Age"] > 100])
print(df[df["Salary"] < 0])
print()

#zad. 4
df.loc[df["Age"] < 0, "Age"] = np.nan
df.loc[df["Age"] > 100] = np.nan
df.loc[df["Salary"] < 0] = np.nan


#zad. 5
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["City"] = df["City"].fillna("Unkown")

#zad. 6
print(df.duplicated())
df = df.drop_duplicates()
print(df.duplicated())
print()

#zad. 7
#1. Bo mediana jest bardziej miarodajna gdyż pokazuje środkową wartość, a średnią łatwo można zaburzyć przez wartości na granicy
#2. Bo to są stringi
#3. Kiedy potrzebuje faktycznie prawdziwych wartości do analizy, wtedy lepiej usunąć wiersz

#zadanie bonusowe

def clean_data(df):
    df.loc[df["Age"] < 0, "Age"] = np.nan
    df.loc[df["Age"] > 100] = np.nan
    df.loc[df["Salary"] < 0] = np.nan

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
    df["City"] = df["City"].fillna("Unkown")
    df = df.drop_duplicates()
    return df