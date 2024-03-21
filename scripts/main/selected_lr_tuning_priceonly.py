import pandas as pd
import torch
import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts')
from data_modelling.CNNs import CNN_config, CNN
from data_modelling.GRUs import GRU_config, GRU
from data_modelling.LSTMs import LSTM_config, LSTM
from data_modelling.RNNs import RNN_config, RNN
from data_modelling.LSTM_GRU import LSTM_GRU_config, LSTM_GRU
from data_modelling.RNN_GRU import RNN_GRU_config, RNN_GRU
from data_modelling.NN_Training import TrainNNModel
from utilities.data_preparation import PrepareModellingData

# This is done for 500 epochs and no early stopping implemented.

lr_parameters_df = pd.read_csv('/home/ubuntu/Masters_Thesis/results/final_results/parameter_tuning/selected_lr_output_price_only_forecast_7.csv')

# This file should contain processed, sentiment past, price past and forecast values. Additionally, it should have the sentiment feature name and model name

model_config = {'CNN': [CNN, CNN_config], 'GRU': [GRU, GRU_config], 'LSTM': [LSTM, LSTM_config], 'RNN': [RNN, RNN_config], 'LSTM_GRU': [LSTM_GRU, LSTM_GRU_config], 'RNN_GRU': [RNN_GRU, RNN_GRU_config]}

def make_config(config, X, y, metric, sentiment_past, price_past, processed, lr):
    config['input_dimension'] = X.shape[1]
    config['hidden_dimension'] = 512
    config['output_dimension'] = y.shape[1]
    config['dropout'] = 0.1
    config['optimizer'] = 'Adam'
    config['sentiment_past'] = sentiment_past
    config['price_past'] = price_past
    config['metric'] = metric
    config['label'] = 'close'
    config['processed'] = processed
    config['model_type'] = 'regression'
    config['eta'] = lr
    return config

frequency = '30T'
label = 'close'
sentiment_past = -1
price_past_list = [15, 30]
time_forecast = 7

def make_config(config, X, y, metric, sentiment_past, price_past, processed, lr):
    config['input_dimension'] = X.shape[1]
    config['hidden_dimension'] = 512
    config['output_dimension'] = y.shape[1]
    config['dropout'] = 0.1
    config['optimizer'] = 'Adam'
    config['sentiment_past'] = sentiment_past
    config['price_past'] = price_past
    config['metric'] = metric
    config['label'] = 'close'
    config['processed'] = processed
    config['model_type'] = 'regression'
    config['eta'] = lr
    return config


if __name__ == "__main__":
    try:
        data_final = []
        for price_past in price_past_list:
            for network_name, [network, config] in model_config.items():
                data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=1, weighted_score='user_weighted')
                data = data_dict[frequency][0]
                processed = False
                X, y = PrepareModellingData(sentiment_past = sentiment_past, price_past = price_past, time_forecast = time_forecast).get_modelling_data_and_output_for_regression(data, None, frequency, label)
                lr = lr_parameters_df.loc[(lr_parameters_df['network_name'] == network_name) & (lr_parameters_df['sentiment_past'] == sentiment_past) & (lr_parameters_df['price_past'] == price_past) & (lr_parameters_df['forecast'] == time_forecast), 'lr'].iloc[0]
                print(f'{network_name} {lr} working...')
                trial_config = make_config(config, X, y, None, sentiment_past, price_past, processed, lr)
                train_class = TrainNNModel(X, y, 64)
                val_mse, val_mae, val_mape, model = train_class.run_training_regression(network, trial_config)
                path = f'/home/ubuntu/Masters_Thesis/results/final_models/{network_name}_price_only_SPT_{sentiment_past}_{price_past}_{time_forecast}.pt'
                torch.save(model, path)

                testloader = train_class.testloader
                yscaler = train_class.y_scaler
                mse_loss, test_mae, test_mape = train_class.evaluate_regression_model(model, testloader, yscaler)
                data_final.append({'network_name': network_name, 'price_past': price_past, 'lr': lr, 'test_mse_loss': mse_loss, "test_mae": test_mae, 'test_mape': test_mape, 'val_loss': val_mse, 'val_mae': val_mae, 'val_mape': val_mape})
    finally:          
        data_final = pd.DataFrame(data_final)
        data_final.to_csv(f'/home/ubuntu/Masters_Thesis/results/final_results/final_model_results/Final_model_output_price_only_forecast_{time_forecast}.csv', index = False)
    
    