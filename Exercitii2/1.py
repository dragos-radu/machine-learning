import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_test)

u_labels = np.unique(y_test)
aligned_labels = np.zeros_like(y_pred)

for i in u_labels:
    cluster = np.where(y_pred == i)[0]
    most_common_true_label = np.bincount(y_test[cluster]).argmax()
    aligned_labels[cluster] = most_common_true_label

accuracy = accuracy_score(y_test, aligned_labels)
conf_matrix = confusion_matrix(y_test, aligned_labels)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion matrix:\n")
print(conf_matrix)

class_representations = {}
for i in range(4):
    cluster_mask = (kmeans.labels_ == i)
    cluster_labels = y_train[cluster_mask]
    most_common = np.argmax(np.bincount(cluster_labels))
    most_common_percentage = (np.bincount(cluster_labels)[most_common] / cluster_labels.size) * 100
    class_representations[i] = (most_common, most_common_percentage)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=200, edgecolor='k')

plt.title('K-Means Clustering pentru setul de date Iris')

legend_labels = [f'Cluster {i} - Class {iris.target_names[class_representations[i][0]]} ({class_representations[i][1]:.2f}%)' for i in class_representations]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters")

plt.show()
