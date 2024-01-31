import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

faces_data = fetch_olivetti_faces()
images = faces_data.images
labels = faces_data.target

n_samples, h, w = images.shape
images_flat = images.reshape((n_samples, h * w))
images_flat = images_flat / 255.0
n_components = 100
pca = PCA(n_components=n_components)
images_compressed = pca.fit_transform(images_flat)

images_reconstructed = pca.inverse_transform(images_compressed)

n_images = 5
plt.figure(figsize=(10, 4))
for i in range(n_images):
    # Imaginea originală
    plt.subplot(2, n_images, i + 1)
    plt.imshow(images[i])
    plt.title(f'Original {i + 1}')
    plt.axis('off')

    # Imaginea reconstruită
    plt.subplot(2, n_images, i + 1 + n_images)
    plt.imshow(images_reconstructed[i].reshape((h, w)), cmap='gray')
    plt.title(f'Reconstructed {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
