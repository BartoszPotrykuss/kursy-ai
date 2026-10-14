import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    "Age": [
        22, 25, 28, 35, 42, 45, 52, 58,
        23, 27, 31, 38, 41, 49, 55, 60,
        24, 29, 33, 36, 44, 47, 51, 57,
        21, 26, 30, 34, 40, 46, 53, 62
    ],
    "Annual Income": [
        25, 30, 35, 45, 50, 55, 60, 65,
        70, 75, 80, 85, 90, 95, 100, 105,
        110, 115, 120, 125, 130, 135, 140, 145,
        40, 48, 58, 68, 78, 88, 98, 108
    ],
    "Purchased": [
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1
    ]
})

df.head()

X = df[["Age", "Annual Income"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000) #KNeighborsClassifier(n_neighbors=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_pred, y_test)}")

"""
plt.scatter(df["Age"], df["Annual Income"], c=df["Purchased"])
plt.xlabel("Wiek")
plt.ylabel("Roczny przychód")
plt.show()
"""

age_range = np.arange(21, 63)
income_range = np.arange(25, 145)
age_grid, income_grid = np.meshgrid(age_range, income_range)

print(age_grid.shape)
print(income_grid.shape)

grid_points = np.column_stack((age_grid.ravel(), income_grid.ravel()))

print(grid_points.shape)

grid_points_scaled = scaler.transform(grid_points)

predictions = model.predict(grid_points_scaled)

print(predictions)

predictions_grid = predictions.reshape(120, 42)

"""
plt.contourf(age_grid, income_grid, predictions_grid, alpha=0.3)
plt.scatter(df["Age"], df["Annual Income"], c=df["Purchased"], edgecolors="black")
plt.xlabel("Wiek")
plt.ylabel("Roczny przychód")
plt.title("KNN — Decision Boundary")
plt.show()
"""

print(model.coef_)
print(model.intercept_)