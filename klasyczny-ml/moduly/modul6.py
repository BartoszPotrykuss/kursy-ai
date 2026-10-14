import pandas as pd

employees = pd.DataFrame({
    "EmployeeID":[1,2,3,4,5],
    "Name":["Anna","Jan","Ola","Piotr","Kasia"],
    "DepartmentID":[101,102,101,103,999],
    "Salary":[7000,6000,7500,8000,6500]
})

departments = pd.DataFrame({
    "DepartmentID":[101,102,103],
    "Department":["IT","HR","Finance"]
})

#zad. 1
print(employees.merge(
    departments,
    on="DepartmentID"
))

#zad. 2
print(employees.merge(
    departments,
    on="DepartmentID",
    how="left"
))

#zad. 3
print(employees.merge(
    departments,
    on="DepartmentID",
    how="right"
))

#zad. 4

print(employees.merge(
    departments,
    on="DepartmentID",
    how="outer"
))

#zad. 5
#Kasia, bo nie ma działu o id 999

#zad. 6
print(employees.merge(
    departments,
    on="DepartmentID"
).groupby("Department")["Salary"].mean())

#zad. 7
team_a = pd.DataFrame({
    "Name":["Adam","Bartek"]
})

team_b = pd.DataFrame({
    "Name":["Celina","Dominik"]
})

print(pd.concat([team_a, team_b]))

#zad. 8
names = pd.DataFrame({
    "Name":["Anna","Jan","Ola"]
})

ages = pd.DataFrame({
    "Age":[20,25,19]
})

print(pd.concat([names, ages], axis=1))

# Tabela Employees
employees = pd.DataFrame({
    "pracownik": [
        "Anna Kowalska",
        "Jan Nowak",
        "Piotr Wiśniewski",
        "Katarzyna Zielińska",
        "Michał Lewandowski",
        "Agnieszka Kamińska",
        "Tomasz Wójcik",
        "Ewa Mazur",
        "Paweł Kaczmarek",
        "Magdalena Dąbrowska"
    ],
    "pensja": [
        6500,
        8200,
        7800,
        9100,
        5600,
        7300,
        6900,
        8800,
        6100,
        9700
    ],
    "id_dzialu": [
        101,
        102,
        101,
        103,
        104,
        102,
        101,
        103,
        104,
        102
    ]
})

# Tabela Departments
departments = pd.DataFrame({
    "id_dzialu": [101, 102, 103, 104],
    "nazwa_dzialu": [
        "IT",
        "Sprzedaż",
        "Marketing",
        "HR"
    ]
})

print(employees)
print()
print(departments)

#mini projekt
print(employees.merge(
    departments,
    on="id_dzialu"
).groupby("nazwa_dzialu").agg(
   ilepracownikow=("pracownik", "count")
))

print(employees.merge(
    departments,
    on="id_dzialu"
).groupby("nazwa_dzialu").agg(
    srednia_pensja=("pensja", "mean")
))

print(employees.merge(
    departments,
    on="id_dzialu"
).groupby("nazwa_dzialu").agg(
    srednia_pensja=("pensja", "mean")
).sort_values("srednia_pensja", ascending=False).head(1))

print(employees.merge(
    departments,
    on="id_dzialu"
).isna().sum().sum())