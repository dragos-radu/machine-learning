from matplotlib import pyplot as plt
import pandas as pd

data = {
    "An": [2010, 2011, 2012, 2013, 2014],
    "Vanzari": [500, 600, 750, 800, 900]
}

df = pd.DataFrame(data)

df.plot(x='An', y='Vanzari', marker='o', linestyle='-', color='b', label='Vanzari')

plt.xlabel('An')
plt.ylabel('Vanzari')
plt.title('Grafic linie - vanzari in functie de An')

plt.legend()
plt.grid(True)

plt.show()
