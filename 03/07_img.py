import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import  image as mpimg

img = mpimg.imread("figure_1.png")
plt.imshow(img)

plt.title("Imagine")

plt.savefig("Exemplu_salvat.png", format='png')

plt.show()

