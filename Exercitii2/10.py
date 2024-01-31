import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

wine = load_wine()
data = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])

dbscan = DBSCAN(eps=10, min_samples=5)
data['cluster'] = dbscan.fit_predict(data.drop(columns=['target']))

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse} => RMSE: {np.sqrt(mse)}')
coefficients = regression_model.coef_
intercept = regression_model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept (constant): {intercept}')

r_squared = regression_model.score(X_test, y_test)
print(f'R-squared: {r_squared}')

feature_importance = pd.Series(regression_model.coef_, index=X.columns)
feature_importance = feature_importance.abs().sort_values(ascending=False)
print(f'Feature Importance:\n{feature_importance}')

