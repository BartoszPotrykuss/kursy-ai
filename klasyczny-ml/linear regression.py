import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print(df.head())
print(df.shape)
print(df.describe())
print(df.isna().sum())
print(df.info())
print(df.columns)

"""
plt.scatter(df["MedInc"], df["MedHouseVal"])
plt.ylabel("Mediana wartości domu")
plt.xlabel("Mediana zarobków")
plt.title("Zarobki a wartość domu")
plt.show()

plt.scatter(df["Population"], df["MedHouseVal"])
plt.ylabel("Mediana wartości domu")
plt.xlabel("Populacja")
plt.title("Zarobki a populacja")
plt.show()
"""

features = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
]

X = df[features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
print(f"R^2: {r2_score(y_test, y_pred)}")
print()
print(f"Coef: {model.coef_}")
print(f"Intercept: {model.intercept_}")

residuals = y_test - y_pred

print(y_pred)
print(residuals)

"""
plt.scatter(y_pred, y_test)
plt.xlabel("Predicted")
plt.ylabel("Test")
plt.title("Predicted vs test")
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Idealne dopasowanie (x=y)")
plt.show()

plt.scatter(y_pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals")
plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="y = 0")
plt.show()

ridge1_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1))
])

ridge1000_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1000))
])

ridge1_pipeline.fit(X_train, y_train)
y_predR1 = ridge1_pipeline.predict(X_test)

ridge1000_pipeline.fit(X_train, y_train)
y_predR1000 = ridge1000_pipeline.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
print(f"R^2: {r2_score(y_test, y_pred)}")
print()
print(f"Coef: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print()
print("Ridge - alpha=1")
print(f"MAE: {mean_absolute_error(y_test, y_predR1)}")
print(f"MSE: {mean_squared_error(y_test, y_predR1)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_predR1)}")
print(f"R^2: {r2_score(y_test, y_predR1)}")
print()
print(f"Coef: {ridge1_pipeline.named_steps['model'].coef_}")
print()
print("Ridge - alpha=1000")
print(f"MAE: {mean_absolute_error(y_test, y_predR1000)}")
print(f"MSE: {mean_squared_error(y_test, y_predR1000)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_predR1000)}")
print(f"R^2: {r2_score(y_test, y_predR1000)}")
print()
print(f"Coef: {ridge1000_pipeline.named_steps['model'].coef_}")

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print()
y_pred_train = rf.predict(X_train)

print("TRAIN")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train)}")
print(f"RMSE: {root_mean_squared_error(y_train, y_pred_train)}")
print(f"R²: {r2_score(y_train, y_pred_train)}")

print()

print("TEST")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred_rf)}")
print(f"R²: {r2_score(y_test, y_pred_rf)}")

"""
"""
rf1 = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

rf2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf3 = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

rf1.fit(X_train, y_train)
rf2.fit(X_train, y_train)
rf3.fit(X_train, y_train)
y_pred_rf1 = rf1.predict(X_test)
y_pred_rf2 = rf2.predict(X_test)
y_pred_rf3 = rf3.predict(X_test)
y_pred_rf1_train = rf1.predict(X_train)
y_pred_rf2_train = rf2.predict(X_train)
y_pred_rf3_train = rf3.predict(X_train)

print("RF1")
print("TRENING")
print(f"R²: {r2_score(y_train, y_pred_rf1_train)}")
print("TEST")
print(f"R²: {r2_score(y_test, y_pred_rf1)}")
print()
print("RF2")
print("TRENING")
print(f"R²: {r2_score(y_train, y_pred_rf2_train)}")
print("TEST")
print(f"R²: {r2_score(y_test, y_pred_rf2)}")
print()
print("RF3")
print("TRENING")
print(f"R²: {r2_score(y_train, y_pred_rf3_train)}")
print("TEST")
print(f"R²: {r2_score(y_test, y_pred_rf3)}")
print()

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

print(r2_score(y_test, y_pred))


print()
configs = [
    (0.01, 100),
    (0.01, 500),
    (0.01, 1000),
    (0.1, 100),
    (0.1, 500),
    (1.0, 100),
]

for lr, n in configs:

    model = GradientBoostingRegressor(
        n_estimators=n,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(
        f"lr={lr}, n_estimators={n} → "
        f"R²={r2_score(y_test, y_pred):.4f}, "
        f"RMSE={root_mean_squared_error(y_test, y_pred):.4f}"
    )

depths = [1, 2, 3, 5, 10]

for depth in depths:

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"max_depth={depth}")
    print(f"TRAIN R²: {r2_score(y_train, y_pred_train):.4f}")
    print(f"TEST  R²: {r2_score(y_test, y_pred_test):.4f}")
    print()
    
"""

param_grid = {
    "n_estimators": [100, 500],
    "learning_rate": [0.01, 0.1, 1],
    "max_depth": [1, 3, 5, None]
}

gridSearch = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring="r2",
    n_jobs=-1,
)

gridSearch.fit(X_train, y_train)
print(gridSearch.best_score_)
print(gridSearch.best_params_)
y_pred_gs = gridSearch.best_estimator_.predict(X_test)
print(r2_score(y_test, y_pred_gs))

