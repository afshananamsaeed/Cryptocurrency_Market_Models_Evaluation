import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class GradientBoostingModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def get_model_results_regression(self):
        self.get_test_train_split()
        model, _ = GradientBoostingModel.fit_gboost_regression_model(self.X_train, self.y_train)
        mae, mse, rmse, mape, r2 = GradientBoostingModel.get_model_analysis_regression(model, self.X_test, self.y_test)
        return mae, mse, rmse, mape, r2
    
    def get_model_results_classification(self):
        self.get_test_train_split()
        model, _ = GradientBoostingModel.fit_gboost_classification_model(self.X_train, self.y_train)
        accuracy, f1 = GradientBoostingModel.get_model_analysis_classification(model, self.X_test, self.y_test)
        return accuracy, f1

    def get_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=False)
        self.X_train, self.X_test = GradientBoostingModel.standardize_data(self.X_train, self.X_test)
        
    @staticmethod
    def standardize_data(X_train, X_test):
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        train = xscaler.fit_transform(X_train)
        test = xscaler.transform(X_test)
        return train, test

    @staticmethod
    def fit_gboost_regression_model(X_train, y_train):
        print("training_xgboost_regression")
        params = {
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'learning_rate': [1, 0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300, 400, 500],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_features': ['sqrt', 'log2']
        }
        gboost_cv = RandomizedSearchCV(estimator=GradientBoostingRegressor(), param_distributions=params, cv=5, verbose=0, n_jobs=-1, scoring='r2', error_score='raise')
        gboost_cv.fit(X_train, np.ravel(y_train))
        gb = gboost_cv.best_estimator_
        est_params = gboost_cv.best_params_
        return gb, est_params
    
    @staticmethod
    def fit_gboost_classification_model(X_train, y_train):
        print("training_xgboost_classification")
        params = {
            'learning_rate': [1, 0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300, 400, 500],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_features': ['sqrt', 'log2']
        }
        gboost_cv = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=params, cv=5, verbose=0, n_jobs=-1, scoring='accuracy', error_score='raise')
        gboost_cv.fit(X_train, np.ravel(y_train))
        gb = gboost_cv.best_estimator_
        est_params = gboost_cv.best_params_
        return gb, est_params

    @staticmethod
    def get_model_analysis_regression(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        GradientBoostingModel.plot_output(y_pred, y_test)
        print(f"MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}, R2: {r2}")
        return mae, mse, rmse, mape, r2
    
    @staticmethod
    def get_model_analysis_classification(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, f1
    
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