import random

import pandas as pd
import random
import seaborn as sns
from matplotlib import pyplot as plt

from faker import Faker

fake = Faker()

data = {
    'Data': [], 'Produs': [], 'Cant': [], 'PU': []
}

for _ in range(100):
    data['Data'].append(fake.date_between(start_date='-30d', end_date='today'))
    data['Produs'].append(random.choice(['Produs_A', 'Produs_B', 'Produs_C']))
    data['Cant'].append(random.randint(10, 50))
    data['PU'].append(round(random.uniform(20, 100), 2))

df1 = pd.DataFrame(data)

# df.to_csv('vanzari.csv', index=False)

values = pd.read_csv('vanzari.csv')

df = pd.DataFrame(values)

df['Venituri'] = df['Cant'] * df['PU']

plt.figure(figsize=(12,6))
total_vanz_zilnice = df.groupby('Data')['Venituri'].sum().reset_index()
print(total_vanz_zilnice)

sns.lineplot(x='Data', y='Venituri', data=total_vanz_zilnice)

plt.title("Ev Vanzari zilnice")
plt.xlabel("Data")
plt.ylabel("Vanzari")

plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['PU'], bins=20, kde=True, color='skyblue')
plt.xlabel("Pret u")
plt.ylabel("Nr prod")
plt.show()


plt.figure(figsize=(10,6))

sns.scatterplot(x='PU', y='Cant', data=df, hue='Produs', palette="viridis")

plt.title("relatia produs cantitate vanduta")
plt.xlabel("Pret unitar")
plt.ylabel("Cantitate Vanduta")

plt.legend(title="Produs")
plt.show()

plt.figure(figsize=(10,6))

import numpy as np

numeric_df = df.select_dtypes(include=[np.number])  # This will select only numeric columns
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

plt.title("Mat corelatie")
plt.show()