import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.random.seed(42)

num_samples = 100
education_year = np.random.randint(8, 20, size=(num_samples, 1))
experience_year = np.random.randint(10, 30, size=(num_samples, 1))
income = 2000 + 100 * education_year + 50 * experience_year + np.random.randn(num_samples, 1) * 500

X = np.hstack((education_year, experience_year))
y = income

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].scatter(y_test, y_pred, color='blue')
ax[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--',
           color='red',
           linewidth=2)

ax[0].set_xlabel("Venit real")
ax[0].set_ylabel("Venit prezis")
ax[0].set_title("Predictii vs real")

residuals = y_test - y_pred
ax[1].scatter(y_pred, residuals, color='green')
ax[1].axhline(y=0, linestyle='--', color='red', linewidth=2)
ax[1].set_xlabel("Venit Prezis")
ax[1].set_ylabel('Reziduuri')
ax[1].set_title("Reziduuri vs predictii")

plt.show()

print(f"Coef de interceptare (beta0): {model.intercept_}")
print(f'Coef de inclinare (beta1, beta2): {model.coef_}')
print(f'MSE: {mse}')






