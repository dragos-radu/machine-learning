from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
acuratete = accuracy_score(Y_test, predictions)
mat = confusion_matrix(Y_test, predictions)
raport_class = classification_report(Y_test, predictions)

print(acuratete)
print(mat)
print(raport_class)





