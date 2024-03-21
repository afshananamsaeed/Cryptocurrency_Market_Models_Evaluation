import pandas as pd
import os
os.chdir('/home/ubuntu/Masters_Thesis/scripts')
import optuna
from data_modelling.GRUs import GRU_config, GRU
from data_modelling.LSTMs import LSTM_config, LSTM
from data_modelling.RNNs import RNN_config, RNN
from data_modelling.LSTM_GRU import LSTM_GRU_config, LSTM_GRU
from data_modelling.RNN_GRU import RNN_GRU_config, RNN_GRU
from data_modelling.TCNs import TCN_config, TCN
from utilities.data_preparation import PrepareModellingData
from data_modelling.Optuna_Tuning import Objective
optuna.logging.set_verbosity(optuna.logging.WARNING)


model_config = {'GRU': [GRU, GRU_config], 'LSTM': [LSTM, LSTM_config], 'RNN': [RNN, RNN_config], 'LSTM_GRU': [LSTM_GRU, LSTM_GRU_config], 'RNN_GRU': [RNN_GRU, RNN_GRU_config], 'TCN': [TCN, TCN_config]}

metric = 'bert_nonprocessed_data_sentiment_score'
label = 'close'
frequency = '30T'
data_dict = PrepareModellingData.load_data_wo_bots(dataset_no=1, weighted_score='user_weighted')
data = data_dict[frequency][0]


def run_checks(sentiment, price, forecast):
    print(f"Starting {sentiment}, {price}, {forecast}")
    result = []
    for model_type in model_config:
        print(f"Starting {model_type}")
        network = model_config[model_type][0]
        config = model_config[model_type][1]
        config['model_type'] = 'regression'
        
        X, y = PrepareModellingData(sentiment_past = sentiment, price_past = price, time_forecast = forecast).get_modelling_data_and_output_for_regression(data, metric, frequency, label)
        
        config['input_dimension'] = X.shape[1]
        config['hidden_dimension'] = 512
        config['output_dimension'] = y.shape[1]
        config['dropout'] = 0.1
        config['optimizer'] = 'Adam'
        config['metric'] = metric
        config['label'] = label
        
        objective = Objective(network, config, X, y)
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize"
        )
        study.optimize(objective, n_trials=100, callbacks=[objective.callback])
        
        # get best params
        trial = study.best_trial
        for _, value in trial.params.items():
            lr = value
            
        print(f"Completed {model_type}")
        
        result.append({'model': model_type, 'sentiment': sentiment, 'price': price, 'forecast':forecast, 'lr': lr})
    
    return result

if __name__ == "__main__":
    result1 = run_checks(sentiment = 15, price = 15, forecast =0)
    result2 = run_checks(sentiment = 15, price = 15, forecast = 7)
    result3 = run_checks(sentiment = -1, price = 15, forecast = 0)
    result4 = run_checks(sentiment = -1, price = 15, forecast = 7)
    
    result = result1+result2+result3+result4
    result = pd.DataFrame(result)
    result.to_csv('/mnt/model_output_nn/lr_checks_result.csv', index = False)
    