from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def align_clusters(y_true, y_pred):
    unique_labels = np.unique(y_true)
    aligned_labels = np.zeros_like(y_pred)
    for k in unique_labels:
        match_cluster = np.where(y_pred == k)[0]
        most_common_true_label = np.bincount(y_true[match_cluster]).argmax()
        aligned_labels[match_cluster] = most_common_true_label
    return aligned_labels


def calculate_class_representation_in_clusters(y_true, clusters, num_clusters=3):
    class_representations = {}
    for i in range(num_clusters):
        cluster_mask = (clusters == i)
        cluster_labels = y_true[cluster_mask]
        most_common = np.argmax(np.bincount(cluster_labels))
        most_common_percentage = (np.bincount(cluster_labels)[most_common] / cluster_labels.size) * 100
        class_representations[i] = (most_common, most_common_percentage)
    return class_representations

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42,  n_init=10)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_test)

labels_aligned = align_clusters(y_test, y_pred)

accuracy = accuracy_score(y_test, labels_aligned)
conf_matrix = confusion_matrix(y_test, labels_aligned)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion matrix:\n")
print(conf_matrix)

class_representations = calculate_class_representation_in_clusters(y_train, kmeans.labels_)
plt.figure(figsize=(10, 6))

scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=200, edgecolor='k')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering pentru setul de date Iris folosind sepal-width si sepal-length')

legend_labels = [f'Cluster {i} - Class {iris.target_names[class_representations[i][0]]} ({class_representations[i][1]:.2f}%)' for i in class_representations]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Clusters")
plt.show()