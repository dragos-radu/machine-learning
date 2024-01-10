import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.random.seed(42)

num_samples = 100

X = np.random.rand(num_samples, 1) * 100

y = 5 * X + np.random.randn(num_samples, 1) * 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

plt.scatter(X_test, y_test, color='black', label='lin med')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Line mod')
plt.xlabel("House area (sq. meters)")
plt.ylabel("House price")
plt.legend()
plt.show()

r2 = r2_score(y_test, y_pred)
print(f"Coef interceptare: {model.intercept_[0]}")
print(f"Coef inclinare: {model.coef_[0][0]}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

