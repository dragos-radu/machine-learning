import pandas as pd

from matplotlib import pyplot as plt

data = {
    "An": [2010, 2011, 2012, 2013, 2014],
    "Vanzari": [500, 600, 750, 800, 900]
}

df = pd.DataFrame(data)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,8))

df.plot(x='An', y='Vanzari', ax=axs[0], marker='o', linestyle="--", color='b', label='Vanzari')
axs[0].set_title("Grafic linie - Vanzari")

df.plot.bar(x='An',y='Vanzari', ax=axs[1], color='g', alpha=0.7, label="Vanzari (bara)")
axs[1].set_title("Grafic bara - Vanzari")


plt.tight_layout()

plt.show()
