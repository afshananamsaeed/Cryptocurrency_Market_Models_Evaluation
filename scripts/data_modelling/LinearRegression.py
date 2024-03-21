import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class LinearRegressionModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def get_model_results_regression(self):
        self.get_test_train_split()
        model, _ = LinearRegressionModel.fit_model(self.X_train, self.y_train)
        mae, mse, rmse, mape, r2 = LinearRegressionModel.get_model_analysis_regression(model, self.X_test, self.y_test)
        return mae, mse, rmse, mape, r2

    def get_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=False)
        self.X_train, self.X_test = LinearRegressionModel.standardize_data(self.X_train, self.X_test)
        
    @staticmethod
    def standardize_data(X_train, X_test):
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        train = xscaler.fit_transform(X_train)
        test = xscaler.transform(X_test)
        return train, test

    @staticmethod
    def fit_model(X_train, y_train):
        lr = LinearRegression()
        parameters = {"fit_intercept": [True, False],
                    "positive": [True, False]
                    }
        grid = GridSearchCV(estimator=lr, param_grid = parameters, cv = 5, n_jobs=-1)
        grid.fit(X_train, y_train)
        lr = grid.best_estimator_
        est_params = grid.best_params_
        return lr, est_params
    
    @staticmethod
    def get_model_analysis_regression(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        LinearRegressionModel.plot_output(y_pred, y_test)
        print(f"MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, R2: {r2}")
        return mae, mse, rmse, mape, r2
    
    @staticmethod
    def plot_output(y_pred, y_actual):
        plt.figure(figsize=(8, 6))
        iterator_values = np.arange(len(y_actual))
        plt.plot(iterator_values, y_actual, linestyle='-', color='blue', linewidth = 1, label='Actual')
        plt.plot(iterator_values, y_pred, linestyle='-', color='red', linewidth = 1, label='Predicted')

        # Adding labels and title
        plt.xlabel('Iterator')
        plt.ylabel('Values')
        plt.title('Actual vs. Predicted Values')
        plt.legend()

        # Display the plot
        plt.show()


class LassoRegressionModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def get_model_results_regression(self):
        self.get_test_train_split()
        model, _ = LassoRegressionModel.fit_model(self.X_train, self.y_train)
        mae, mse, rmse, mape, r2 = LinearRegressionModel.get_model_analysis_regression(model, self.X_test, self.y_test)
        return mae, mse, rmse, mape, r2

    def get_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=False)
        self.X_train, self.X_test = LinearRegressionModel.standardize_data(self.X_train, self.X_test)
        
    @staticmethod
    def standardize_data(X_train, X_test):
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        train = xscaler.fit_transform(X_train)
        test = xscaler.transform(X_test)
        return train, test

    @staticmethod
    def fit_model(X_train, y_train):
        params = {
            'alpha': [0.0001,0.001,0.01,0.1,1,10,100,1000]
        }
        lr_cv = GridSearchCV(estimator=Lasso(), param_grid=params, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        lr_cv.fit(X_train, y_train)
        lr = lr_cv.best_estimator_
        est_params = lr_cv.best_params_
        return lr, est_params
    
    @staticmethod
    def get_model_analysis_regression(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        LinearRegressionModel.plot_output(y_pred, y_test)
        print(f"MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, R2: {r2}")
        return mae, mse, rmse, mape, r2
    
class RidgeRegressionModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def get_model_results_regression(self):
        self.get_test_train_split()
        model, _ = RidgeRegressionModel.fit_model(self.X_train, self.y_train)
        mae, mse, rmse, mape, r2 = LinearRegressionModel.get_model_analysis_regression(model, self.X_test, self.y_test)
        return mae, mse, rmse, mape, r2

    def get_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=False)
        self.X_train, self.X_test = LinearRegressionModel.standardize_data(self.X_train, self.X_test)
        
    @staticmethod
    def standardize_data(X_train, X_test):
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        train = xscaler.fit_transform(X_train)
        test = xscaler.transform(X_test)
        return train, test

    @staticmethod
    def fit_model(X_train, y_train):
        params = {
            'alpha': [0.0001,0.001,0.01,0.1,1,10,100,1000]
        }
        lr_cv = GridSearchCV(estimator=Ridge(), param_grid=params, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        lr_cv.fit(X_train, y_train)
        lr = lr_cv.best_estimator_
        est_params = lr_cv.best_params_
        return lr, est_params
    
    @staticmethod
    def get_model_analysis_regression(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        LinearRegressionModel.plot_output(y_pred, y_test)
        print(f"MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, R2: {r2}")
        return mae, mse, rmse, mape, r2
    
class ElasticNetRegressionModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_model_results_regression(self):
        self.get_test_train_split()
        model, _ = ElasticNetRegressionModel.fit_model(self.X_train, self.y_train)
        mae, mse, rmse, mape, r2 = LinearRegressionModel.get_model_analysis_regression(model, self.X_test, self.y_test)
        return mae, mse, rmse, mape, r2

    def get_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=False)
        self.X_train, self.X_test = LinearRegressionModel.standardize_data(self.X_train, self.X_test)
        
    @staticmethod
    def standardize_data(X_train, X_test):
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        train = xscaler.fit_transform(X_train)
        test = xscaler.transform(X_test)
        return train, test

    @staticmethod
    def fit_model(X_train, y_train):
        params = {
            'alpha': [0.0001,0.001,0.01,0.1,1,10,100,1000],
            'l1_ratio': np.arange(0, 1, 0.1)
        }
        lr_cv = GridSearchCV(estimator=ElasticNet(), param_grid=params, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        lr_cv.fit(X_train, y_train)
        lr = lr_cv.best_estimator_
        est_params = lr_cv.best_params_
        return lr, est_params

    @staticmethod
    def get_model_analysis_regression(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        LinearRegressionModel.plot_output(y_pred, y_test)
        print(f"MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, R2: {r2}")
        return mae, mse, rmse, mape, r2