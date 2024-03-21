import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
import pandas as pd
from utilities.data_preparation import PrepareModellingData
from data_modelling.LinearRegression import LinearRegressionModel, LassoRegressionModel, RidgeRegressionModel, ElasticNetRegressionModel
from data_modelling.LinearSVM import LinearSVMModel
from data_modelling.RandomForest import RandomForestModel
from data_modelling.XGBoost import GradientBoostingModel

dependent_regression_labels = ['close', 'close_t+1', 'close_t+2', 'close_t+3', 'close_t+7']
dependent_regression_percentage_labels = ['delta_close_t+0', 'delta_close_t+1', 'delta_close_t+2', 'delta_close_t+3', 'delta_close_t+7']
dependent_categorical_labels = ['delta_close_t+0_class', 'delta_close_t+1_class', 'delta_close_t+2_class', 'delta_close_t+3_class', 'delta_close_t+7_class']
model_dict = {'lr': LinearRegressionModel, 'lasso': LassoRegressionModel, 'ridge': RidgeRegressionModel, 'elasticnet': ElasticNetRegressionModel, 'linearsvm': LinearSVMModel, 'rf': RandomForestModel, 'gb': GradientBoostingModel}

regression_models = ['lr', 'lasso', 'ridge', 'elasticnet', 'linearsvm', 'rf', 'gb']
classification_models = ['linearsvm', 'rf', 'gb']

class GetModelledOutput:
    
    def run_modelling(self):
        
        regression_results = []
        regression_percentage_results = []
        classification_results = []
        
        try:
            for model_type in regression_models:
                for label in dependent_regression_labels:
                    label_result = self.get_regression_modelling_per_frequency(model_type, label)
                    regression_results.extend(label_result)
                print(f"Completed {model_type} regression")
                    
                for label in dependent_regression_percentage_labels:
                    label_result = self.get_regression_percentage_modelling_per_frequency(model_type, label)
                    regression_percentage_results.extend(label_result)
                print(f"Completed {model_type} regression percentage")
                    
            for model_type in classification_models:
                for label in dependent_categorical_labels:
                    label_results = self.get_classification_modelling_per_frequency(model_type, label)
                    classification_results.extend(label_results)
                print(f"Completed {model_type} classification")
        finally:
            print("Saving files...")
            regression = pd.DataFrame(regression_results)
            regression_percentage = pd.DataFrame(regression_percentage_results)
            classification = pd.DataFrame(classification_results)
            
            regression.to_csv(f'/home/ubuntu/Masters Thesis/data/model_results/models/model_output_regression_rf_gb.csv', index=False)
            regression_percentage.to_csv(f'/home/ubuntu/Masters Thesis/data/model_results/models/model_output_regression_percentage_rf_gb.csv', index=False)
            classification.to_csv(f'/home/ubuntu/Masters Thesis/data/model_results/models/model_output_classification_gb.csv', index=False)

    def get_regression_modelling_per_frequency(self, model_type, label):
        results_all_frequencies = []
        data_dict = PrepareModellingData.load_data_wo_bots()
        frequencies = data_dict.keys()
        for frequency in frequencies:
            result = GetModelledOutput.get_regression_modelling_for_data(data_dict[frequency], model_type, frequency, label)
            results_all_frequencies.extend(result)
        return results_all_frequencies
    
    def get_regression_percentage_modelling_per_frequency(self, model_type, label):
        results_all_frequencies = []
        data_dict = PrepareModellingData.load_data_wo_bots()
        frequencies = data_dict.keys()
        for frequency in frequencies:
            result = GetModelledOutput.get_regression_percentage_modelling_for_data(data_dict[frequency], model_type, frequency, label)
            results_all_frequencies.extend(result)
        return results_all_frequencies
    
    def get_classification_modelling_per_frequency(self, model_type, label):
        results_all_frequencies = []
        data_dict = PrepareModellingData.load_data_wo_bots()
        frequencies = data_dict.keys()
        for frequency in frequencies:
            result = GetModelledOutput.get_classification_modelling_for_data(data_dict[frequency], model_type, frequency, label)
            results_all_frequencies.extend(result)
        return results_all_frequencies
    
    @staticmethod
    def get_regression_modelling_for_data(data_list, model_type, frequency, label):
        result = []
        for data in data_list:
            features = data.columns.tolist()
            features.remove('date')
            for metric in features:
                X, y = PrepareModellingData().get_modelling_data_and_output_for_regression(data, metric, frequency, label)
                ml = GetModelledOutput.get_model_type(model_type, X, y)
                mae, mse, rmse, mape, r2 = ml.get_model_results_regression()
                result.append({'model': model_type, 'metric': metric, 'frequency': frequency, 'label': label, 'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2})
        return result
    
    @staticmethod
    def get_regression_percentage_modelling_for_data(data_list, model_type, frequency, label):
        result = []
        for data in data_list:
            features = data.columns.tolist()
            features.remove('date')
            for metric in features:
                X, y = PrepareModellingData().get_modelling_data_and_output_for_regression_percentage(data, metric, frequency, label)
                ml = GetModelledOutput.get_model_type(model_type, X, y)
                mae, mse, rmse, mape, r2 = ml.get_model_results_regression()
                result.append({'model': model_type, 'metric': metric, 'frequency': frequency, 'label': label, 'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2})
        return result
    
    @staticmethod
    def get_classification_modelling_for_data(data_list, model_type, frequency, label):
        result = []
        for data in data_list:
            features = data.columns.tolist()
            features.remove('date')
            for metric in features:
                prd = PrepareModellingData()
                X, y = prd.get_modelling_data_and_output_for_classification(data, metric, frequency, label)
                ml = GetModelledOutput.get_model_type(model_type, X, y)
                accuracy, f1 = ml.get_model_results_classification()
                result.append({'model': model_type, 'metric': metric, 'frequency': frequency, 'label': label, 'accuracy': accuracy, 'f1': f1})
        return result

    @staticmethod
    def get_model_type(model_type, X, y):
        model_class = model_dict[model_type]
        return model_class(X, y)
    