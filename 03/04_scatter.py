import pandas as pd

from matplotlib import pyplot as plt

data = {
    "Nota_mate":[75, 80, 60, 90, 85],
    "Nota_eng": [85, 70, 95, 80, 75]
}

df = pd.DataFrame(data)

df.plot.scatter(x='Nota_mate', y='Nota_eng', color='green', marker='o')

plt.xlabel('Nota mate')
plt.ylabel('Nota eng')

plt.title('Relatia intre notele la mate si eng')

plt.show()
