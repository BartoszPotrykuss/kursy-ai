import pandas as pd

df = pd.DataFrame({
    "Name": [
        "Anna","Jan","Ola","Piotr",
        "Kasia","Tomek","Ewa","Marek"
    ],

    "City": [
        "Kraków",
        "Kraków",
        "Warszawa",
        "Warszawa",
        "Gdańsk",
        "Gdańsk",
        "Kraków",
        "Warszawa"
    ],

    "Department":[
        "IT",
        "HR",
        "IT",
        "HR",
        "IT",
        "HR",
        "IT",
        "IT"
    ],

    "Salary":[
        7000,
        5500,
        8000,
        6000,
        7500,
        5000,
        9000,
        8500
    ],

    "Age":[
        24,
        31,
        28,
        35,
        26,
        30,
        29,
        27
    ]
})


#zad. 1
print(df.groupby("City")["Salary"].mean())
print(df.groupby("City")["Salary"].min())
print(df.groupby("City")["Salary"].max())
print()

#zad. 2
print(df.groupby("City")["Name"].count())
print(df.groupby("City")["Salary"].sum())

#zad. 3
print(df.groupby("Department")["Age"].mean())
print(df.groupby("Department")["Salary"].mean())

#zad. 4
print(df.groupby(["City", "Department"])["Salary"].mean())

#zad. 5
print(df.groupby("City")["Salary"].agg(["mean", "median", "min", "max", "std"]))

#zad. 6
print(df.groupby("City")["Salary"].mean().sort_values(ascending=False))

#zad. 7
print(df.groupby("City")["Salary"].agg(["mean", "median", "min", "max", "std"]).reset_index())

#mini projekt
#1. Warszawa
#2. IT
#3. Kraków i Warszawa
#4. 26.8
#5. Warszawa

