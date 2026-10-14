import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import seaborn as sns

#etap 1

df = pd.read_csv("heart.csv")
pd.set_option('display.max_columns', None)
print(df.sample(5))
print(df.shape)
print(df.info())
#print(df.describe())


#etap 2
df["ageRange"] = pd.cut(df["age"], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90])
print(df.groupby("ageRange").agg(
    ilosc_osob=("age", "count")
))

"""
plt.hist(df["age"], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90])
plt.ylabel("Ilość osób")
plt.xlabel("Wiek")
plt.title("Rozkład wieku")
plt.show()
"""

zdrowi = len(df.loc[df["target"] == 0])
chorzy = len(df.loc[df["target"] == 1])
print(f"Procent zdrowych: {zdrowi/len(df)}")
print(f"Procent chorych: {chorzy/len(df)}")

"""
plt.hist([df.loc[df["target"] == 0, "age"], df.loc[df["target"] == 1, "age"]], label=["Zdrowy", "Chory"], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90])
plt.legend()
plt.ylabel("Ilosc osob")
plt.xlabel("Wiek")
plt.title("Rozkład wieku z podziałem ze względu na obecność choroby")
plt.show()
"""
print("Ile jest mężczyzn ogółem procentowo: ", len(df[df["sex"] == 1])/len(df))
print("Chorzy mężczyźni", len(df[(df["sex"] == 1) & (df["target"] == 1)])/chorzy)
print("Zdrowi mężczyźni", len(df[(df["sex"] == 1) & (df["target"] == 0)])/zdrowi)
print()
print()

#etap 4
columnsX = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

X = df[columnsX]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

#etap 5
#Logistic Regression
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Wyniki dla Logistic Regression")
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred)}")
cechy_wagi = pd.Series(
    model.coef_.flatten(),
    index=X.columns
)
print(cechy_wagi)
print()

y_pred_proba = model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

"""
plt.plot(fpr, tpr)
plt.xlabel("Fpr")
plt.ylabel("Tpr")
plt.title("ROC Curve")
plt.show()
"""

cm = confusion_matrix(y_test, y_pred)
etykiety = ["zdrowy", "chory"]
"""
sns.heatmap(
    data2=cm,
    annot=True,
    fmt='d',
    xticklabels=etykiety,
    yticklabels=etykiety
)
plt.xlabel("Przewidzana klasa")
plt.ylabel("Prawdziwa klasa")
plt.show()
"""

#precision vs recall dla LogisticRegression
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print()

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Wyniki dla Decision Tree")
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred)}")
print(f"Feature importance:")
importance = pd.Series(
    tree.feature_importances_,
    index=X.columns
)
print(importance)
print()


plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["Zdrowy", "Chory"],
    filled=True
)
plt.show()


forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
y_pred_proba = forest.predict_proba(X_test)

print("Wyniki dla Random Forest")
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred)}")
print()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

"""
plt.plot(fpr, tpr)
plt.xlabel("Fpr")
plt.ylabel("Tpr")
plt.title("ROC Curve")
plt.show()
"""
#Etap 6
print()
print("Cross validation dla LogisticRegression: ")
scores = cross_val_score(
    LogisticRegression(max_iter=10000),
    X,
    y,
    cv=5
)
print("Średni wynik: ", scores.mean())
print("Odchyelenie standardowe: ", scores.std())


#Etap 7
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_params_)
best_tree = grid.best_estimator_
y_pred = best_tree.predict(X_test)
print("Accuracy dla grid", accuracy_score(y_test, y_pred))

#Etap 8
#overfitting dla LogisticRegression
print(f"Wynik na zbiorze treningowym {model.score(X_train, y_train):.2%}")
print(f"Wynik na zbiorze testowym {model.score(X_test, y_test):.2%}")

print()
#overfitting dla drzewa
print(f"Wynik na zbiorze treningowym {tree.score(X_train, y_train):.2%}")
print(f"Wynik na zbiorze testowym {tree.score(X_test, y_test):.2%}")
