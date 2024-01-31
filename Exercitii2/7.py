from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

housing_data = fetch_california_housing()

df = pd.DataFrame(data=housing_data.data, columns=housing_data.feature_names)
df['MedHouseVal'] = housing_data.target
full_df_string = df.to_string()

# Print the full DataFrame item
print(full_df_string)
X = housing_data.data
y = housing_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Performanța modelului de regresie liniară:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

random_house_features = [[2.3886, 16.0, 5.254717, 1.162264, 1387.0, 2.616981, 39.37, -121.24]]
predicted_price = linear_reg.predict(random_house_features)
print("Caracteristici aleatorii ale locuinței:")
print(random_house_features)
print("\nPrețul prezis al locuinței:")
print(predicted_price)

