import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_curve, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
pd.set_option('display.max_columns', None)
#print(df.sample(5))

#print(pd.get_dummies(df["Sex"]).sample(5))
#print(pd.get_dummies(df["Embarked"]).sample(5))

median_age = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(median_age)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = df["FamilySize"] == 1

df_encoded = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"]
)

print(df_encoded.head())


X = df_encoded[
    [
        "Pclass",
        "Age",
        "Fare",
        "FamilySize",
        "Sex_female",
        "Sex_male",
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S"
    ]
]

y = df_encoded["Survived"]

"""
print(X.shape)
print(y.shape)

print(X.head())
print(y.head())
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(y_test.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)

y_proba = model.predict_proba(X_test)
#print(y_proba[0])

print("Accuracy: ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ", cm)


"""
etykiety = ["Nie przeżył", "Przeżył"]
plt.figure(figsize=(6, 5))
sns.heatmap(
    data2=cm,
    annot=True,
    fmt="d",
    xticklabels=etykiety,
    yticklabels=etykiety

)
plt.xlabel("Przewidzana klasa")
plt.ylabel("Prawdziwa klasa")
plt.show()
"""

print(model.coef_)
print(model.intercept_)

cechy_wagi = pd.Series(model.coef_.flatten(), index=X.columns)
print(cechy_wagi)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

#0.3
print()
print("0.3: ")
threshold = 0.3
y_pred_03 = (y_proba[:, 1] >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_03)
precision = precision_score(y_test, y_pred_03)
recall = recall_score(y_test, y_pred_03)
f1 = f1_score(y_test, y_pred_03)

print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

#0.7
print()
print("0.7: ")
threshold = 0.7
y_pred_07 = (y_proba[:, 1] >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_07)
precision = precision_score(y_test, y_pred_07)
recall = recall_score(y_test, y_pred_07)
f1 = f1_score(y_test, y_pred_07)

print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

print(len(fpr))
print(len(tpr))
print(len(thresholds))

print(fpr[:10])
print(tpr[:10])
print(thresholds[:10])


"""
plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.show()
"""
print(roc_auc_score(y_test, y_proba[:,1]))

print(classification_report(y_test, y_pred))


#drzewko!!

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Train accuracy:", tree.score(X_train, y_train))
print("Test accuracy:", tree.score(X_test, y_test))

for depth in [1, 2, 3, 4, 5, 6, 8, 10, None]:
    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

    if depth != None:
        print(
            f"depth={depth:<4} ",
            f"train={train_acc:.3f} ",
            f"test={test_acc:.3f}"
        )
    else:
        print(
            f"depth=none ",
            f"train={train_acc:.3f} ",
            f"test={test_acc:.3f}"
        )


tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)

tree.fit(X_train, y_train)

"""
plt.figure(figsize=(20, 10))

plot_tree(
    tree,
    feature_names=X_train.columns,
    class_names=["Nie przeżył", "Przeżył"],
    filled=True
)

plt.show()
"""

importance = pd.Series(
    tree.feature_importances_,
    index=X_train.columns
)

print(importance.sort_values(ascending=False))


"""
plt.barh(
    importance.index,
    importance.values
)

plt.xlabel("Feature Importance")
plt.title("Decision Tree Feature Importance")
plt.show()
"""

forest = RandomForestClassifier(
    random_state=42
)

forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

print("Train: ", forest.score(X_train, y_train))
print("Test: ", forest.score(X_test, y_test))

print(classification_report(y_test, y_pred_forest))

for n in [1, 5, 10, 50, 100, 200, 500]:
    forest = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    forest.fit(X_train, y_train)

    print(
        f"{n:>3} trees  "
        f"train={forest.score(X_train, y_train):.3f}  "
        f"test={forest.score(X_test, y_test):.3f}"
    )


importance = pd.Series(
    forest.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importance)


#cross-validationn

scores = cross_val_score(
    LogisticRegression(max_iter=1000),
    X,
    y,
    cv=5
)

print(scores)
print(scores.mean())
print(scores.std())


tree = DecisionTreeClassifier(random_state=42)

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
print(grid.best_params_)
print(grid.best_score_)
best_tree = grid.best_estimator_
y_pred = best_tree.predict(X_test)

print(accuracy_score(y_test, y_pred))

results = pd.DataFrame(grid.cv_results_)

print(
    results[
        [
            "param_max_depth",
            "param_min_samples_split",
            "param_criterion",
            "mean_test_score",
            "std_test_score"
        ]
    ].sort_values("mean_test_score", ascending=False).head(10)
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

