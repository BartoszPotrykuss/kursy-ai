import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name":["Anna","Jan","Ola","Piotr","Kasia"],
    "Age":[20,25,np.nan,22,20],
    "City":["Kraków","Gdańsk","Kraków","Poznań","Gdańsk"],
    "Salary":[5000,7000,6000,6500,np.nan]
})

#zad. 1
print()
print(df.head(3))
print(df.tail(2))
print(df.sample(2))

#zad. 2
print()
print(df.shape)
print(df.size)
print(df.columns)

#zad. 3
print()
print(df.info())
print(df.describe())

#zad. 4
print()
print(df["City"].unique())
print(df["City"].nunique())
print(df["City"].value_counts())

#zad. 5
print()
print(df.isna().sum())
print(df.isna().sum().sum())


#zad. 6
#1. Age i City
#2. Float, przez to że mamy wartości nan
#3. 3
#4. 6125

#dodatkowe: W salary mogą być wartości nan i wtedy one sie nie liczą do liczenia więc dlatego mogą być różne wartości, bo len policzy wszystkie wiersze