import pandas as pd

#1
df = pd.read_csv('01_iris.csv')
print(f"Forma datelor: {df.shape}")
print(f"Tipul datelor: {df.dtypes}")
print(f"Primele 3 randuri:\n {df.head(3)}")

#2
print(f"Cheile datelor: {df.keys()}")
num_rows, num_columns = df.shape
print(f"Randuri: {num_rows}; Coloane: {num_columns}")
print(df.describe())

#3
print(f"\nNumarul observatiilor: {len(df)}")
print(f"Numarul valorilor lipsa: {df.isnull().sum().sum()}")
print(f"Numarul de valori NaN: {df.isna().sum().sum()}\n")

#4
detalii_statistice = df.describe()
print(detalii_statistice)

#5
grupuri_specii = df.groupby('Species').describe()
print(grupuri_specii)

#6
df_iris = df.drop('Id', axis=1)
print(df.drop('Id', axis=1))

#7
print(f"Primele 4 celule: \n {df.loc[:3]}")

#8
import matplotlib.pyplot as plt
import seaborn as sns

medie = df_iris.groupby("Species").mean()
sns.set(style="whitegrid")
medie.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Mediile pentru fiecare specie (Iris)')
plt.xlabel('Caracteristici')
plt.ylabel('Medie')
plt.show()

#9
frecventa_specii = df_iris['Species'].value_counts()

# Afișează un grafic cu bare pentru frecvența speciilor
plt.figure(figsize=(8, 6))
frecventa_specii.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Frecventa Speciilor')
plt.xlabel('Specie')
plt.ylabel('Frecventa')
plt.xticks(rotation=0)  # Pentru a afișa etichetele pe orizontală
plt.show()

#10
plt.figure(figsize=(8, 8))
plt.pie(frecventa_specii, labels=frecventa_specii.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'orange', 'green'])
plt.title('Frecventa Speciilor')
plt.show()

#11
plt.figure(figsize=(8, 6))
plt.scatter(df_iris['SepalLengthCm'], df_iris['SepalWidthCm'], c='blue', alpha=0.7)
plt.title('Relația dintre lungimea și latimea sepalei')
plt.xlabel('Lungimea sepalei (cm)')
plt.ylabel('Latimea sepalei (cm)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='SepalLengthCm', y='SepalWidthCm', data=df_iris, hue='Species', palette='viridis')
plt.title('Relatia dintre lungimea și latimea petalei')
plt.xlabel('Lungimea Petalei (cm)')
plt.ylabel('Latimea Petalei (cm)')
plt.show()

#12
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', data=df_iris, hue='Species', palette='viridis', alpha=0.7)
plt.title('Relația dintre lungimea și Latimea petalei')
plt.xlabel('Lungimea petalei (cm)')
plt.ylabel('Latimea petalei (cm)')
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='PetalLengthCm', y='PetalWidthCm', data=df_iris, hue='Species', palette='viridis')
plt.title('Relația dintre Lungimea și Latimea Petalei')
plt.xlabel('Lungimea Petalei (cm)')
plt.ylabel('Latimea petalei (cm)')
plt.show()

#13
plt.subplot(2, 2, 1)
sns.histplot(df_iris['SepalLengthCm'], kde=True, color='blue')
plt.title('Distributia lunigimii sepalei')

plt.subplot(2, 2, 2)
sns.histplot(df_iris['SepalWidthCm'], kde=True, color='orange')
plt.title('Distributia latimii sepalei')

plt.subplot(2, 2, 3)
sns.histplot(df_iris['PetalLengthCm'], kde=True, color='green')
plt.title('Distributia lungimii petalei')

plt.subplot(2, 2, 4)
sns.histplot(df_iris['PetalWidthCm'], kde=True, color='red')
plt.title('Distributia latimii petalei')

plt.tight_layout()
plt.show()

#14
plt.figure(figsize=(8, 6))
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=df_iris, kind='scatter', color='blue', marginal_kws=dict(bins=25, fill=False))
plt.suptitle('Joint plot: llungimea vs latimea sepalei')
plt.show()

#15
plt.figure(figsize=(8, 6))
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=df_iris, kind='hex', color='blue', marginal_kws=dict(bins=25, fill=False))
plt.suptitle('Hexbin joint plot: lungimea vs lattimea sepalei')
plt.show()

#16
plt.figure(figsize=(8, 6))
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=df_iris, kind='scatter', marginal_kws=dict(bins=25, fill=False))
plt.suptitle('Scatterplot cu estimare a densitatii: lungimea vs latimea sepalei')
plt.show()

#17
sns.set(style="ticks")
sns.pairplot(df_iris, hue="Species", markers=["o", "s", "D"])
plt.suptitle('Pair Plot')
plt.show()

#18
df_num = df_iris.drop('Species', axis=1)
corelatii = df_num.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corelatii, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Harta de Căldura a corelatiei ')
plt.show()

#19
X = df_iris.iloc[:, :4]
y = df['Species']
print("Atribute:")
print(X.head())
print("\nEtichete:")
print(y.head())


#20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Antrenare:", X_train.shape, y_train.shape)
print(X_train, y_train)
print("Testare:", X_test.shape, y_test.shape)
print(X_test, y_test)

#21
from sklearn.preprocessing import LabelEncoder

labelE = LabelEncoder()
df_iris['Numere'] = labelE.fit_transform(df_iris['Species'])
df_le = df_iris.drop('Species', axis=1)

X = df_iris.iloc[:, :4]
y = df_iris['Numere']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Antrenare:", X_train.shape, y_train.shape)
print(X_train, y_train)
print("Testare:", X_test.shape, y_test.shape)
print(X_test, y_test)

#22
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cls_report = classification_report(y_test, y_pred)

print("Precizia modelului KNN:", accuracy)
print(f"Raportul de clasificare:\n {cls_report}")

#23
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acuratetea modelului KNN:", accuracy)

#24
ks = []
for k in range(1, 11):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    ks.append(accuracy)
    print(f"K = {k}: Acc = {accuracy:.4f}")

#25
plt.plot(ks, marker='o')
plt.title('Accuratetea pentru valori diferite ale k in KNN')
plt.xlabel('Numarul de vecini (k)')
plt.ylabel('Acuratete')
plt.show()

#26
plt.figure(figsize=(8, 6))
plt.plot(range(1,11), ks, marker='o')
plt.title('Valori ale lui k versus acuratete în KNN')
plt.xlabel('Numarul de vecini (k)')
plt.ylabel('Acuratete')
plt.show()

#27
detalii_specii = df.groupby('Species').describe()

print("Detalii statistice pentru 'Iris-setosa':")
print(detalii_specii.loc['Iris-setosa'])

print("\nDetalii statistice pentru 'Iris-versicolor':")
print(detalii_specii.loc['Iris-versicolor'])

print("\nDetalii statistice pentru 'Iris-virginica':")
print(detalii_specii.loc['Iris-virginica'])

#28
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df_iris, palette='viridis')
plt.title('Grafic de dispersie pentru lungimea si latimea sepalei')
plt.xlabel('lungimea sepalei (cm)')
plt.ylabel('Latimea petalei (cm)')
plt.legend(title='Specie')
plt.show()

#29
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.metrics import r2_score

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

print('Regresie logistica')
print(f'Acuratetea modelului logistic: {accuracy_logistic:.4f}')
print('\nRaport de clasificare:')
print(classification_report(y_test, y_pred_logistic))
print('\nMatrice de confuzie:')
print(confusion_matrix(y_test, y_pred_logistic))


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
y_pred_linear_round = [round(pred) for pred in y_pred_linear]
mse_linear = mean_squared_error(y_test, y_pred_linear_round)
r2 = r2_score(y_test, y_pred)
print()
print('Regresie liniara')
print('\nRaport de clasificare:')
print(classification_report(y_test, y_pred_linear_round))
print(f'Mean Squared Error pentru regresia liniara: {mse_linear:.4f}')
print(f"R2: {r2}")
