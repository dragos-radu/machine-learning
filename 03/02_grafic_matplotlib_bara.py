import pandas as pd

from matplotlib import pyplot as plt

data = {
    "An": [2010, 2011, 2012, 2013, 2014],
    "Vanzari": [500, 600, 750, 800, 900]
}

df = pd.DataFrame(data)

df.plot(x='An', y='Vanzari', kind='bar', color='g', label='Vanzari')

plt.xlabel('An')
plt.ylabel('Vanzari')
plt.title('Grafic bara - vanzari in functie de An')


plt.legend()
plt.grid(axis='y')

plt.show()
