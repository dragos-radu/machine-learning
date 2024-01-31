from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
data = mnist.data.astype(float)
labels = mnist.target.astype(int)

pca = PCA(n_components=2)  # Reducem la 2 componente pentru vizualizare
data_pca = pca.fit_transform(data)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5)
plt.xlabel('Componenta Principală 1')
plt.ylabel('Componenta Principală 2')
plt.title('PCA pe setul de date MNIST')
plt.colorbar(scatter, label='Cifra')
plt.grid(True)
plt.show()
