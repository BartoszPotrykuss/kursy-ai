import pandas as pd

df = pd.DataFrame({
    "Name": ["Anna", "Jan", "Ola", "Piotr", "Kasia", "Tomek", "Zosia"],
    "Age": [20, 25, 19, 22, 20, 31, 27],
    "City": ["Kraków", "Gdańsk", "Kraków", "Poznań", "Gdańsk", "Kraków", "Warszawa"],
    "Salary": [5000, 7000, 6000, 6500, 5500, 9000, 8000]
})

print(df[df["Age"] > 25])
print()
print(df[df["Salary"] >= 7000])
print()
print(df[df["City"] == "Kraków"])
print()
print(df[df["City"].isin(["Kraków", "Warszawa"])])
print()
print(df[df["Age"].between(20, 25)])
print()
print(df[(df["Age"] > 20) & (df["Salary"] > 6000)])
print()
print(df.loc[(df["Age"] > 20) & (df["Salary"] > 6000), ["City", "Salary"]])
print()
print(df.iloc[0:3, [0, 3]])
print()
print(df[df["City"].isin(["Kraków", "Gdańsk"])])
print()
print(df[~(df["City"] == "Kraków") & (df["Age"] < 30)])
print()
print(df.loc[(df["Age"] < 30) & (df["Salary"] >= 6500) & ~(df["City"] == "Kraków"), ["Name", "City", "Salary"]])