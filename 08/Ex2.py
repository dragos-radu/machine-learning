import matplotlib.pyplot as plt
import pandas as pd
from keras.src.datasets import cifar10
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(X_test.shape)

print(y_test.shape, y_train.shape)
classes = np.unique(y_train)
nClass = len(classes)
print(nClass)
print(classes)
plt.figure(figsize=[5, 5])
labels_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
plt.subplot(121)
curr_img = np.reshape(X_train[0], (32, 32, 3))
plt.imshow(curr_img)
print(plt.title(labels_dict[y_train[0][0]]))

plt.subplot(122)
curr_img = np.reshape(X_test[0], (32, 32, 3))
plt.imshow(curr_img)
print(plt.title(labels_dict[y_test[0][0]]))
plt.show()

print(np.min(X_train), np.max(X_train))
X_train = X_train/255
print(np.min(X_train), np.max(X_train))
print(X_train.shape)

x_train_flat = X_train.reshape(-1, 3072)
feat_cols = [f"pixel {i}" for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat, columns=feat_cols)
df_cifar['label'] = y_train
print(f"Marime df {df_cifar.shape}")
print(df_cifar.head())

pca_cifar = PCA(n_components=2)
pComp = pca_cifar.fit_transform(df_cifar.iloc[:, :-1])
cifar_df = pd.DataFrame(data=pComp, columns=['p1', 'p2'])
cifar_df['y'] = y_train
print(f"variatia {pca_cifar.explained_variance_ratio_}")


plt.figure(figsize=(16,10))
sns.scatterplot(x='p1',
                y='p2',
                hue='y',
                palette=sns.color_palette('hls',10),
                data=cifar_df,
                legend='full',
                alpha=0.3)
plt.show()


