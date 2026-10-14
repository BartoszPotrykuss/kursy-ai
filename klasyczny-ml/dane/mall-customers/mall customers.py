import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


df = pd.read_csv('Mall_Customers.csv')

print(df.shape)
print(df.info())
print(df.head(5))
print(df.columns)
print(df.isna().sum().sum())
print(df.describe())
print()

df_genres = df.groupby("Genre").agg(
    liczbaOsob=("Genre", "count")
)

print(df_genres)

"""
plt.bar(df_genres.index, df_genres["liczbaOsob"], color=["blue", "orange"])
plt.xlabel("Płeć")
plt.ylabel("Liczba osób")
plt.title("Liczba osób według płci")
plt.show()

plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
plt.xlabel("Roczny przychód")
plt.ylabel("Wskaźnik wydatków")
plt.title("Zależność między rocznym przychodem a wskaźnikiem wydatków")
plt.show()

plt.scatter(df["Age"], df["Spending Score (1-100)"])
plt.xlabel("Wiek")
plt.ylabel("Wskaźnik wydatków")
plt.title("Zależność między wiekiem a wskaźnikiem wydatków")
plt.show()

plt.scatter(df["Age"], df["Annual Income (k$)"])
plt.xlabel("Wiek")
plt.ylabel("Roczny przychód")
plt.title("Zależność między wiekiem a rocznym przychodem")
plt.show()
"""

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[0:5])
print(X_scaled.mean())
print(X_scaled.std())
print()

"""
inertias = []

for n in range(2, 11):
    kmean = KMeans(n_clusters=n)
    kmean.fit(X_scaled)
    inertias.append(kmean.inertia_)

print(inertias)


plt.plot(range(2, 11), inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Wartość inertia")
plt.title("Elbow method")
plt.show()

scores = []

for n in range(2, 11):
    kmean = KMeans(n_clusters=n)
    labels = kmean.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)

print(scores)

plt.plot(range(2, 11), scores)
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("Silhouette score")
plt.show()
"""


kmean = KMeans(n_clusters=5, random_state=42)
df["cluster_5"] = kmean.fit_predict(X_scaled)

print(df.groupby("cluster_5")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2))
print()

"""
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["cluster_5"])
plt.xlabel("Roczny przychód")
plt.ylabel("Wskaźnik wydatków")
plt.title("Zależność między rocznym przychodem a wskaźnikiem wydatków")
plt.show()

plt.scatter(df["Age"], df["Spending Score (1-100)"], c=df["cluster_5"])
plt.xlabel("Wiek")
plt.ylabel("Wskaźnik wydatków")
plt.title("Zależność między wiekiem a wskaźnikiem wydatków")
plt.show()
"""

result = pd.crosstab(df["cluster_5"], df["Genre"], normalize="index") * 100
result = result.round(2)
result.columns.name = None

print(result)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster_5"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA")
plt.show()