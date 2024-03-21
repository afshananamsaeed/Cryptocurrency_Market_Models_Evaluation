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

# lr_parameters_df = pd.read_csv('/mnt/model_output_nn/user_weighted/dataset1/parameter_tuning/selected_lr_output_SPT_30_30_0.csv')
lr_parameters_df = pd.read_csv('/home/ubuntu/Masters_Thesis/results/final_results/parameter_tuning/selected_lr_output_SPT_30_30_0.csv')

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

nonprocessed_features = ['flair_nonprocessed_data_score', 'textblob_nonprocessed_data_polarity_score', 'textblob_nonprocessed_data_subjectivity_score', 'vader_nonprocessed_data_score', 'bert_nonprocessed_data_sentiment_score', 'roberta_nonprocessed_data_sentiment_score']
processed_features = ['flair_processed_data_score', 'textblob_processed_data_polarity_score', 'textblob_processed_data_subjectivity_score', 'vader_processed_data_score', 'bert_processed_data_sentiment_score', 'roberta_processed_data_sentiment_score']

frequency = '30T'
label = 'close'
sentiment_past = 30
price_past = 30
time_forecast = 0

def train_model():
    try:
        data_final_nonprocessed = []
        for feature in nonprocessed_features:
            for network_name, [network, config] in model_config.items():
                data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=1, weighted_score='user_weighted')
                data = data_dict[frequency][0]
                processed = False
                X, y = PrepareModellingData(sentiment_past = sentiment_past, price_past = price_past, time_forecast = time_forecast).get_modelling_data_and_output_for_regression(data, feature, frequency, label)
                lr = lr_parameters_df.loc[(lr_parameters_df['network_name'] == network_name) & (lr_parameters_df['feature'] == feature) & (lr_parameters_df['sentiment_past'] == sentiment_past) & (lr_parameters_df['price_past'] == price_past) & (lr_parameters_df['forecast'] == time_forecast) & (lr_parameters_df['processed'] == processed), 'lr'].iloc[0]
                
                print(f'{network_name} {lr} {feature} working...')
                trial_config = make_config(config, X, y, feature, sentiment_past, price_past, processed, lr)
                train_class = TrainNNModel(X, y, 64)
                val_mse, val_mae, val_mape, model = train_class.run_training_regression(network, trial_config)
                path = f'/home/ubuntu/Masters_Thesis/results/final_models/{network_name}_{feature}_processed_{processed}_SPT_{sentiment_past}_{price_past}_{time_forecast}.pt'
                torch.save(model, path)

                testloader = train_class.testloader
                yscaler = train_class.y_scaler
                mse_loss, test_mae, test_mape = train_class.evaluate_regression_model(model, testloader, yscaler)
                print(network_name, feature)
                print(mse_loss, test_mae, test_mape)
                data_final_nonprocessed.append({'network_name': network_name, 'feature': feature, 'lr': lr, 'test_mse_loss': mse_loss, "test_mae": test_mae, 'test_mape': test_mape, 'val_loss': val_mse, 'val_mae': val_mae, 'val_mape': val_mape})
    finally:  
        print(data_final_nonprocessed)          
        # data_final_nonprocessed = pd.DataFrame(data_final_nonprocessed)
        # data_final_nonprocessed.to_csv(f'/mnt/model_output_nn/user_weighted/dataset1/final_model_results/Final_model_outputs_processed_False_SPT_{sentiment_past}_{price_past}_{time_forecast}.csv', index = False)
    
    try:
        data_final_processed = []
        for feature in processed_features:
            for network_name, [network, config] in model_config.items():
                data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=1, weighted_score='user_weighted')
                data = data_dict[frequency][1]
                processed = True
                X, y = PrepareModellingData(sentiment_past = sentiment_past, price_past = price_past, time_forecast = time_forecast).get_modelling_data_and_output_for_regression(data, feature, frequency, label)
                lr = lr_parameters_df.loc[(lr_parameters_df['network_name'] == network_name) & (lr_parameters_df['feature'] == feature) & (lr_parameters_df['sentiment_past'] == sentiment_past) & (lr_parameters_df['price_past'] == price_past) & (lr_parameters_df['forecast'] == time_forecast) & (lr_parameters_df['processed'] == processed), 'lr'].iloc[0]
                
                print(f'{network_name} {lr} {feature} working...')
                trial_config = make_config(config, X, y, feature, sentiment_past, price_past, processed, lr)
                train_class = TrainNNModel(X, y, 64)
                val_mse, val_mae, val_mape, model = train_class.run_training_regression(network, trial_config)
                # path = f'/home/ubuntu/Masters_Thesis/results/final_models/{network_name}_{feature}_processed_{processed}_SPT_{sentiment_past}_{price_past}_{time_forecast}.pt'
                # torch.save(model, path)
                # train_class.plot_test_output_predictions(model, trial_config)

                testloader = train_class.testloader
                yscaler = train_class.y_scaler
                mse_loss, test_mae, test_mape = train_class.evaluate_regression_model(model, testloader, yscaler)
                print(network_name, feature)
                print(mse_loss, test_mae, test_mape)
                data_final_processed.append({'network_name': network_name, 'feature': feature, 'lr': lr, 'test_mse_loss': mse_loss, "test_mae": test_mae, 'test_mape': test_mape, 'val_loss': val_mse, 'val_mae': val_mae, 'val_mape': val_mape})
    finally:            
        data_final_processed = pd.DataFrame(data_final_processed)
        # data_final_processed.to_csv(f'/mnt/model_output_nn/user_weighted/dataset1/final_model_results/Final_model_outputs_processed_True_SPT_{sentiment_past}_{price_past}_{time_forecast}.csv', index = False)
        # data_final_processed.to_csv(f'/home/ubuntu/Masters_Thesis/results/final_results/final_model_results/LSTM_single_output_processed_True_SPT_{sentiment_past}_{price_past}_{time_forecast}.csv', index = False)
        
if __name__ == "__main__":
    train_model()