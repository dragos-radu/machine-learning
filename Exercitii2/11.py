import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

digits = load_digits()
data, target = digits.data, digits.target

kmeans = KMeans(n_init=1000, n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

result_df = pd.DataFrame({'PC1': reduced_data[:, 0], 'PC2': reduced_data[:, 1], 'Cluster': clusters, 'Target': target})
print(result_df)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=result_df, palette='Spectral', legend='full', s=30)
plt.title('K-Means -> PCA on the digits dataset')
plt.show()

