import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = data.data
y = data.target

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

X_clustered = np.column_stack((X, clusters))

regression_model = LinearRegression()
regression_model.fit(X_clustered, y)
predictions = regression_model.predict(X_clustered)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustered)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_result = pd.DataFrame({'Cluster': clusters, 'Actual Target': y, 'Predicted Target': predictions})
print(df_result.head())
df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Target': y})
print(df_pca.head())

