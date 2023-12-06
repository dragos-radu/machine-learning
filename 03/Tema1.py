import pandas as pd
import random
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from faker import Faker

fake = Faker('ro_RO')

Neighborhood = [fake.administrative_unit() for _ in range(42)]

data = {
    'Neighborhood': [], 'Price': [], 'Occupancy': [], 'Review_Score': []
}

for _ in range(100):
    data['Neighborhood'].append(random.choice(Neighborhood))
    data['Price'].append(round(random.uniform(70, 200), 2))
    data['Occupancy'].append(random.randint(0, 100))
    data['Review_Score'].append(round(random.uniform(1, 5), 1))

df = pd.DataFrame(data)

total_airbnb_per_region = df.groupby('Neighborhood').size().reset_index(name='AirBnbs')
print(total_airbnb_per_region)
sns.barplot(data=total_airbnb_per_region, x='Neighborhood', y='AirBnbs')
plt.title("Airbnbs per region")
plt.show()

mean_score_per_region = df.groupby('Neighborhood')['Review_Score'].mean().reset_index()
print(mean_score_per_region)
sns.barplot(data=mean_score_per_region, x='Neighborhood', y='Review_Score')
plt.title("Mean score per region")
plt.show()

mean_occupancy_per_region = df.groupby('Neighborhood')['Occupancy'].mean().reset_index()
print(mean_occupancy_per_region)
sns.barplot(data=mean_occupancy_per_region, x='Neighborhood', y='Occupancy')
plt.title("Mean occupancy per region")
plt.show()

mean_price_per_region = df.groupby('Neighborhood')['Price'].mean().reset_index()
print(mean_price_per_region)
sns.barplot(data=mean_price_per_region, x='Neighborhood', y='Price')
plt.title("Mean price per region")
plt.show()

numeric_df = df.select_dtypes(include=[np.number])  # This will select only numeric columns
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice corelatie")
plt.show()
