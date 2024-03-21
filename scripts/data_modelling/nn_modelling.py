import pandas as pd
import numpy as np
import optuna
from data_modelling.CNNs import CNN_config, CNN
from data_modelling.GRUs import GRU_config, GRU
from data_modelling.LSTMs import LSTM_config, LSTM
from data_modelling.RNNs import RNN_config, RNN
from data_modelling.LSTM_GRU import LSTM_GRU_config, LSTM_GRU
from data_modelling.RNN_GRU import RNN_GRU_config, RNN_GRU
from data_modelling.TCNs import TCN_config, TCN
from utilities.data_preparation import PrepareModellingData
from data_modelling.NN_Training import TrainNNModel

dependent_regression_labels = ['close', 'close_t+1', 'close_t+2', 'close_t+3', 'close_t+7']
dependent_regression_percentage_labels = ['delta_close_t+0', 'delta_close_t+1', 'delta_close_t+2', 'delta_close_t+3', 'delta_close_t+7']
dependent_categorical_labels = ['delta_close_t+0_class', 'delta_close_t+1_class', 'delta_close_t+2_class', 'delta_close_t+3_class', 'delta_close_t+7_class']
model_config = {'CNN': [CNN, CNN_config], 'GRU': [GRU, GRU_config], 'LSTM': [LSTM, LSTM_config], 'RNN': [RNN, RNN_config], 'LSTM_GRU': [LSTM_GRU, LSTM_GRU_config], 'RNN_GRU': [RNN_GRU, RNN_GRU_config], 'TCN': [TCN, TCN_config]}

class GetNNModelledOutput:

    def run_modelling(self, dataset = 1, weighted_score = 'user_weighted'):
    
        regression_results = []
        # regression_percentage_results = []
        # classification_results = []
        
        try:
            for model_type in model_config.keys():
                for label in dependent_regression_labels:
                    label_result = self.get_regression_modelling_per_frequency(model_type, label, dataset, weighted_score)
                    regression_results.extend(label_result)
                print(f"Completed {model_type} regression")
                
                # for label in dependent_regression_percentage_labels:
                #     label_result = self.get_regression_percentage_modelling_per_frequency(model_type, label, dataset)
                #     regression_percentage_results.extend(label_result)
                # print(f"Completed {model_type} regression percentage")
                
                # for label in dependent_categorical_labels:
                #     label_results = self.get_classification_modelling_per_frequency(model_type, label, dataset)
                #     classification_results.extend(label_results)
                # print(f"Completed {model_type} classification")
        finally:
            print("Saving files...")
            regression = pd.DataFrame(regression_results)
            # regression_percentage = pd.DataFrame(regression_percentage_results)
            # classification = pd.DataFrame(classification_results)
            
            regression.to_csv(f'/mnt/model_output_nn/{weighted_score}/dataset{dataset}/model_output_regression_nn_timepast15.csv', index=False)
            # regression_percentage.to_csv(f'/home/ubuntu/Masters Thesis/data/model_results/models/model_output_regression_percentage_nn.csv', index=False)
            # classification.to_csv(f'/home/ubuntu/Masters Thesis/data/model_results/models/model_output_classification_nn.csv', index=False)

    def get_regression_modelling_per_frequency(self, model_type, label, dataset, weighted_score):
        results_all_frequencies = []
        data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=dataset, weighted_score=weighted_score)
        # frequencies = data_dict.keys()
        frequencies = ['30T']
        for frequency in frequencies:
            result = GetNNModelledOutput.get_regression_modelling_for_data(data_dict[frequency], model_type, frequency, label)
            results_all_frequencies.extend(result)
        return results_all_frequencies
    
    def get_regression_percentage_modelling_per_frequency(self, model_type, label, dataset, weighted_score):
        results_all_frequencies = []
        data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=dataset, weighted_score=weighted_score)
        frequencies = data_dict.keys()
        for frequency in frequencies:
            result = GetNNModelledOutput.get_regression_percentage_modelling_for_data(data_dict[frequency], model_type, frequency, label)
            results_all_frequencies.extend(result)
        return results_all_frequencies
    
    def get_classification_modelling_per_frequency(self, model_type, label, dataset, weighted_score):
        results_all_frequencies = []
        data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=dataset, weighted_score=weighted_score)
        frequencies = data_dict.keys()
        for frequency in frequencies:
            result = GetNNModelledOutput.get_classification_modelling_for_data(data_dict[frequency], model_type, frequency, label)
            results_all_frequencies.extend(result)
        return results_all_frequencies
    
    @staticmethod
    def get_regression_modelling_for_data(data_list, model_type, frequency, label):
        result = []
        for data in data_list:
            features = data.columns.tolist()
            features.remove('date')
            for metric in features:
                print(f'{model_type}_{frequency}_{label}_{metric} working.....')
                X, y = PrepareModellingData().get_modelling_data_and_output_for_regression(data, metric, frequency, label)
                network, config = GetNNModelledOutput.get_model(model_type, X, y, regression_type = 'Regression')
                train = TrainNNModel(network, X, y, 64)
                val_mse_loss, mae, mape, _, plt = train.run_training_regression(config)
                plt.savefig(f'/mnt/NN_plots/{model_type}_{frequency}_{label}_{metric}_regression_output.png')
                mean_mse_loss = np.mean(val_mse_loss[1:])
                min_mse_loss = np.min(val_mse_loss)
                mean_mae = np.mean(mae[1:])
                min_mae = np.min(mae)
                mean_mape = np.mean(mape[1:])
                min_mape = np.min(mape)
                result.append({'model': model_type, 'metric': metric, 'frequency': frequency, 'label': label, 'mean_mse_loss': mean_mse_loss, 'min_mse_loss':min_mse_loss, 'mean_mae': mean_mae, 'min_mae': min_mae, 'mean_mape': mean_mape, 'min_mape': min_mape})
        return result
    
    @staticmethod
    def get_regression_percentage_modelling_for_data(data_list, model_type, frequency, label):
        result = []
        for data in data_list:
            features = data.columns.tolist()
            features.remove('date')
            for metric in features:
                X, y = PrepareModellingData().get_modelling_data_and_output_for_regression_percentage(data, metric, frequency, label)
                ml = GetNNModelledOutput.get_model_type(model_type, X, y)
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
                ml = GetNNModelledOutput.get_model_type(model_type, X, y)
                accuracy, f1 = ml.get_model_results_classification()
                result.append({'model': model_type, 'metric': metric, 'frequency': frequency, 'label': label, 'accuracy': accuracy, 'f1': f1})
        return result

    @staticmethod
    def get_model(model_type, X, y, regression_type):
        network = model_config[model_type][0]
        config_file = model_config[model_type][1]
        config_file['input_dimension'] = X.shape[1]
        if regression_type == 'Regression':
            config_file['eta'] = 0.0003
            config_file['output_dimension'] = 1
        elif regression_type == 'Regression_Percentage':
            config_file['eta'] = 0.0001
            config_file['output_dimension'] = 1
        elif regression_type == 'Classification':
            config_file['eta'] = 0.0001
            config_file['output_dimension'] = len(set(y))
        network_class = network(config_file)
        return network_class, config_file