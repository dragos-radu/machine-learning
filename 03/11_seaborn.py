import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


iris = sns.load_dataset('iris')
print(iris)

plt.figure(figsize=(12, 6))

sns.barplot(x='species', y='sepal_length', data = iris)

plt.title('Medie sepal length pe specie')

plt.xlabel('Specie')
plt.ylabel('Medie Sepal')

plt.show()

plt.figure(figsize=(14,8))
sns.violinplot(x='species', y='sepal_length', data=iris, palette='muted')

plt.xlabel('Specie')
plt.ylabel('Medie Sepal')

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(x='sepal_length', y="sepal_width", data=iris, palette='Dark2',hue= 'sepal_length')

plt.title("Relatie width length")

plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')

plt.show()

sns.pairplot(iris, hue='species', palette='husl')

plt.show()


