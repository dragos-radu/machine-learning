
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler

digits = load_digits()
digits_data = digits.data
digits_target = digits.target

labels = np.reshape(digits_target, (1797, 1))
final_digits_data = np.concatenate([digits_data, labels], axis=1)
digits_dataset = pd.DataFrame(final_digits_data)
features = digits.feature_names
features_labels = np.append(features, 'label')
digits_dataset.columns = features_labels

x = digits_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)


pca_digits = PCA(n_components=2)
principalComponents_digits = pca_digits.fit_transform(x)
principal_digits_Df = pd.DataFrame(data=principalComponents_digits, columns=['principal component 1', 'principal component 2'])
principal_digits_Df['y'] = digits_dataset['label']
print(principal_digits_Df.head())
print(f"Variatia pe fiecare componenta: {pca_digits.explained_variance_ratio_}")

plt.figure(figsize=(14, 10))
sns.scatterplot(x='principal component 1', y='principal component 2',
                hue='y',
                palette=sns.color_palette('hls', 10),
                data=principal_digits_Df,
                legend='full',
                alpha=0.3)
plt.show()

num_components = [2, 5, 10, 20, 30, 40, 50]
explained_variances = []
for n in num_components:
    pca = PCA(n_components=n)
    transformed_data = pca.fit_transform(final_digits_data)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

plt.plot(num_components, explained_variances, marker='o')
plt.title('Variance Explained by Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.show()