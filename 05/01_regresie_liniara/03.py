import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

california_dataset = fetch_california_housing()


def show_data(dataset):
    # print(dataset.keys())

    df = pd.DataFrame(dataset.data,
                      columns=dataset.feature_names)

    df["MEDV"] = dataset.targe


def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def todo1(input_data, target_data, percent_training_data):
    X_train, X_predict, y_train, y_predict = train_test_split(input_data,
                                                              target_data,
                                                              test_size=(100-percent_training_data)/100,
                                                              random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_output = model.predict(X_predict)

    pe = get_rmse(predictions=model_output, targets=y_predict)
    t = range(1, len(model_output) + 1)
    print(f'Predictions error (RMSE): {pe}')
    prediction_error = np.sqrt(np.power((model_output - y_predict), 2))
    plt.subplot(211)

    plt.plot(t, y_predict, 'b')
    plt.plot(t, model_output, 'g')
    plt.legend(['target', 'predictions'])
    plt.ylabel("Housing prices")

    plt.title("California Dataset medina housing value")

    plt.subplot(212)

    plt.plot(prediction_error, 'b')
    plt.legend([f"RMSE: {pe}"])

    plt.xlabel("x (samples)")
    plt.ylabel("Prediction error (RMSE)")
    plt.show()

    return pe


def todo2(input_data, target_data):
    percent_training_data_range = range(90, 50, -1)
    pe_vect = []
    for percent_training_data in percent_training_data_range:
        pe_vect.append(todo1(input_data, target_data, percent_training_data))

    print(pe_vect)
    print(min(pe_vect))

    plt.plot(percent_training_data_range, pe_vect, 'b')
    plt.legend(['Prediction error'])
    plt.ylabel("RMSE")
    plt.xlabel("Percent training data")
    plt.show()


input_data = california_dataset.data
target_data = california_dataset.target

#todo2(input_data, target_data)

todo1(input_data, target_data, 69)

