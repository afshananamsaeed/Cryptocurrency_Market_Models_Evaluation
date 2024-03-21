import pandas as pd
import numpy as np
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

# this is done for 100 epochs and early stopping implemented

model_config = {'CNN': [CNN, CNN_config], 'GRU': [GRU, GRU_config], 'LSTM': [LSTM, LSTM_config], 'RNN': [RNN, RNN_config], 'LSTM_GRU': [LSTM_GRU, LSTM_GRU_config], 'RNN_GRU': [RNN_GRU, RNN_GRU_config]}
feature_names = []

frequency = '30T'
label = 'close'
sentiment_past = -1
price_past_list = [5, 15, 30]
time_forecast = 1

val_1 = np.linspace(1e-5, 1e-3, 40).tolist()
val_2 = np.linspace(1e-6, 1e-4, 40).tolist()

lr_models = {'CNN': val_2, 'LSTM': val_1, 'GRU': val_2, 'RNN': val_2, 'LSTM_GRU': val_2, 'RNN_GRU': val_2}

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
                for lr in lr_models[network_name]:
                    print(f'{network_name} {lr} working...')
                    trial_config = make_config(config, X, y, None, sentiment_past, price_past, processed, lr)
                    train_class = TrainNNModel(X, y, 64)
                    val_mse, mae, mape, model = train_class.run_training_regression(network, trial_config)
                    # train_class.plot_test_output_predictions(model, trial_config)

                    testloader = train_class.testloader
                    yscaler = train_class.y_scaler
                    mse_loss, test_mae, test_mape = train_class.evaluate_regression_model(model, testloader, yscaler)
                    data_final.append({'network_name': network_name, 'price_past': price_past, 'lr': lr, 'test_mse_loss': mse_loss, "test_mae": test_mae, 'test_mape': test_mape})
    finally:          
        data_final = pd.DataFrame(data_final)
        data_final.to_csv(f'/mnt/model_output_nn/user_weighted/dataset1/parameter_tuning/output_manual_price_only_forecast_{time_forecast}.csv', index = False)
        