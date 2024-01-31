import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv')
# print(data.head())

df['Total'] = df['Fresh'] + df['Milk'] + df['Grocery'] + df['Frozen'] + df['Detergents_Paper'] + df['Delicassen']

X = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'Total']]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

print(X.describe())

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

