import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
import xgboost

housing = fetch_california_housing(as_frame=True)

df = housing.frame

features = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
]

X = df[features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=3,
    early_stopping_rounds=50,
    random_state=42
)

xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Najlepsza liczba drzew:", xgb.best_iteration)

y_pred = xgb.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")