from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

class LinearSVMModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_model_results_regression(self):
        self.get_test_train_split()
        model, _ = LinearSVMModel.fit_svm_regression_model(self.X_train, self.y_train)
        mae, mse, rmse, mape, r2 = LinearSVMModel.get_model_analysis_regression(model, self.X_test, self.y_test)
        
        return mae, mse, rmse, mape, r2
    
    def get_model_results_classification(self):
        self.get_test_train_split()
        model, _ = LinearSVMModel.fit_svm_classification_model(self.X_train, self.y_train)
        accuracy, f1 = LinearSVMModel.get_model_analysis_classification(model, self.X_test, self.y_test)
        return accuracy, f1
    
    def get_test_train_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42, shuffle=False)
        self.X_train, self.X_test = LinearSVMModel.standardize_data(self.X_train, self.X_test)
        
    @staticmethod
    def standardize_data(X_train, X_test):
        xscaler = MinMaxScaler(feature_range=(-1, 1))
        train = xscaler.fit_transform(X_train)
        test = xscaler.transform(X_test)
        return train, test

    @staticmethod
    def fit_svm_regression_model(X_train, y_train):
        params = {
            'C': [1, 10, 100, 1000],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
        }
        svm_cv = GridSearchCV(estimator=LinearSVR(), param_grid=params, cv=5, verbose=0, n_jobs=-1)
        svm_cv.fit(X_train, np.ravel(y_train))
        svm = svm_cv.best_estimator_
        est_params = svm_cv.best_params_
        return svm, est_params

    @staticmethod
    def fit_svm_classification_model(X_train, y_train):
        params = {
            'C': [1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'class_weight':['balanced', None]
        }
        svm_cv = GridSearchCV(estimator=LinearSVC(), param_grid=params, cv=5, verbose=0, n_jobs=-1)
        svm_cv.fit(X_train, y_train.ravel())
        svm = svm_cv.best_estimator_
        est_params = svm_cv.best_params_
        return svm, est_params

    @staticmethod
    def get_model_analysis_regression(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        LinearSVMModel.plot_output(y_pred, y_test)
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