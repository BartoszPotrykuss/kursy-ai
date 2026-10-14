import pandas as pd

df = pd.DataFrame({
    "Name": [
        "Anna","Jan","Ola","Piotr",
        "Kasia","Tomek","Ewa","Marek"
    ],

    "City": [
        "Kraków","Kraków","Warszawa","Warszawa",
        "Gdańsk","Gdańsk","Kraków","Warszawa"
    ],

    "Department":[
        "IT","HR","IT","HR",
        "IT","HR","IT","IT"
    ],

    "Salary":[
        7000,5500,8000,6000,
        7500,5000,9000,8500
    ],

    "Age":[
        24,31,28,35,
        26,30,29,27
    ]
})

#zad. 1
print(pd.pivot_table(
    df,
    index="City",
    values="Salary"
))

#zad. 2
print(pd.pivot_table(
    df,
    index="City",
    columns="Department",
    values="Salary"
))

#zad. 3
print(pd.pivot_table(
    df,
    index="City",
    values="Salary",
    aggfunc="max"
))

#zad. 4
print(pd.pivot_table(
    df,
    index="City",
    values="Salary",
    aggfunc=["mean", "min", "max"]
))

#zad. 5
print(pd.pivot_table(
    df,
    index="City",
    values=["Salary", "Age"]
))

#zad. 6
print(pd.pivot_table(
    df,
    index="City",
    values=["Salary", "Age"],
    fill_value=0
))

#zad. 7
print(pd.pivot_table(
    df,
    index="City",
    values=["Salary", "Age"],
    fill_value=0,
    margins=True
))

#mini projekt
#1. Warszawa
#2. Gdansk
#3. 28.75
#4. Warszawa