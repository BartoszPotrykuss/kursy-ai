import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


df = pd.DataFrame({
    "Age": [
        19, 21, 24, 26, 29, 31, 34, 37, 41, 45,
        48, 52, 55, 59, 23, 27, 32, 36, 43, 50,
        20, 25, 30, 35, 40, 46, 51, 57, 62, 28,
        33, 39, 44, 49, 54, 60, 22, 38, 47, 53
    ],
    "AnnualIncome": [
        22, 25, 28, 31, 35, 38, 42, 45, 50, 55,
        60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
        110, 115, 120, 125, 130, 135, 140, 145, 150, 52,
        68, 78, 88, 98, 108, 118, 32, 72, 102, 128
    ],
    "MonthlySpend": [
        120, 150, 180, 200, 220, 250, 270, 300, 320, 350,
        380, 400, 430, 450, 500, 520, 550, 580, 600, 620,
        650, 680, 700, 720, 750, 780, 800, 850, 900, 340,
        410, 470, 530, 590, 640, 690, 230, 490, 610, 760
    ],
    "VisitsPerMonth": [
        1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
        5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
        10, 10, 11, 11, 12, 12, 13, 13, 14, 4,
        6, 7, 8, 9, 10, 11, 2, 6, 9, 12
    ],
    "Membership": [
        "No", "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes",
        "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "Yes",
        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No",
        "No", "Yes", "Yes", "Yes", "Yes", "No", "No", "Yes", "Yes", "Yes"
    ],
    "Device": [
        "Mobile", "Mobile", "Desktop", "Mobile", "Desktop", "Mobile",
        "Desktop", "Mobile", "Desktop", "Mobile", "Desktop", "Mobile",
        "Desktop", "Mobile", "Mobile", "Desktop", "Mobile", "Desktop",
        "Mobile", "Desktop", "Mobile", "Desktop", "Mobile", "Desktop",
        "Mobile", "Desktop", "Mobile", "Desktop", "Mobile", "Desktop",
        "Mobile", "Desktop", "Mobile", "Desktop", "Mobile", "Desktop",
        "Mobile", "Desktop", "Mobile", "Desktop"
    ],
    "WillBuyNextMonth": [
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1
    ]
})



print(df.shape)
print(df.describe())
print(df.info())
print(df.isna().sum())
print()

print(len(df))
print(len(df[df["WillBuyNextMonth"] == 1]))
print()

X = df.drop(columns="WillBuyNextMonth")
y = df["WillBuyNextMonth"]

numericalFeatures = [
    'Age', 'AnnualIncome', 'MonthlySpend', 'VisitsPerMonth'
]

categoricalFeatures = [
    'Membership', 'Device'
]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numericalFeatures),
    ("cat", categorical_pipeline, categoricalFeatures)
])

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

cv_logistic_results = cross_validate(
    logistic_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print("Wyniki CV dla LogisticRegression")
print(cv_logistic_results["test_f1"])
print(cv_logistic_results["test_f1"].mean())
print()

randomForest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])
randomForest_pipeline.fit(X_train, y_train)

cv_rf_results = cross_validate(
    randomForest_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"]
)

print("Wyniki CV dla random forest")
print(cv_rf_results["test_f1"])
print(cv_rf_results["test_f1"].mean())

#Wybierzmy Random Forest do GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    randomForest_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print()
print("GridSearchCV: ")
print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Najlepsze f1: {grid_search.best_score_}")
best_forest = grid_search.best_estimator_
y_pred = best_forest.predict(X_test)

print()
print(f"Końcowe wyniki na teście dla najlepszego modelu: {f1_score(y_test, y_pred)}")

print()
print("Feature importance: ")
rf_model = randomForest_pipeline.named_steps["model"]
feature_names = randomForest_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pd.Series(rf_model.feature_importances_, index=feature_names)
print(importances)