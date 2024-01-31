import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    "Id": np.arange(1, 101),  # Id-uri de la 1 la 100
    "RI": np.random.uniform(1.5, 1.6, 100),
    "Na": np.random.uniform(12, 14, 100),
    "Mg": np.random.uniform(3.5, 4.5, 100),
    "Al": np.random.uniform(1, 2, 100),
    "Si": np.random.uniform(70, 75, 100),
    "K": np.random.uniform(0, 1, 100),
    "Ca": np.random.uniform(7, 9, 100),
    "Ba": np.random.uniform(0, 0.5, 100),
    "Fe": np.random.uniform(0, 0.1, 100),
    "Type": np.random.randint(1, 8, 100)
}

df = pd.DataFrame(data)
# print(df)
df.to_csv("Glass.csv", index=False)

data = pd.read_csv("Glass.csv")
X = data.drop(columns=["Id", "Type"])
y = data["Type"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8, align='center')
plt.title('Varianța explicativă pentru fiecare componentă principală')
plt.xlabel('Componenta Principală')
plt.ylabel('Varianța Explicativă')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='r')
plt.title('Varianța explicativă cumulativă')
plt.xlabel('Numărul de componente principale')
plt.ylabel('Varianța Explicativă Cumulativă')
plt.tight_layout()
plt.show()

num_samples = 100
random_components = np.random.randn(num_samples, 9)
new_samples_pca = pca.inverse_transform(random_components)
new_samples = scaler.inverse_transform(new_samples_pca)
new_samples_df = pd.DataFrame(data=new_samples)
new_samples_df.insert(0, 'ID', range(1, num_samples + 1))
new_samples_df.to_csv("Glass.csv", header=False, index=False, mode='a')

