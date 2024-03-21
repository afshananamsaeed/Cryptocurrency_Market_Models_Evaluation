import torch
import numpy as np
from data_modelling.NN_Training import TrainNNModel

class Objective:
    def __init__(self, network, config, X, y):
        self.best_booster = None
        self._booster = None
        self.test_mae = None
        self.test_mse_loss = None
        self.test_mape = None
        self.train_class = TrainNNModel(X, y, 64)
        self.network = network
        self.config = config
        self.model_type = config['model_type']
        self.model_name = self.network.__name__
        self.suggested_values = set()
        self.max_lr = 1e-4 if self.model_name!='LSTM' else 1e-3
        self.min_lr = 1e-5 if self.model_name!='LSTM' else 1e-4
        self.lr_values = np.linspace(self.min_lr, self.max_lr, 30).tolist()
    
    def __call__(self, trial):
        # model_name = self.network.__name__
        # if model_name == 'LSTM':
        #     (min, max) = (1e-4, 1e-3)
        # else:
        #     (min, max) = (1e-5, 1e-4)
        # self.lr_values = np.linspace(min, max, 20).tolist()
        
        trial_config = self.config
        # lr = self.unique_suggest_float(trial, min, max)
        lr = trial.suggest_categorical('lr', self.lr_values)
        trial_config['eta'] = lr
        
        if trial_config['model_type'] == 'classification':
            network = self.network
            val_loss, model = self.train_class.train_classification(network, trial_config)
            self._booster = model
            return val_loss
        else:
            trial_network = self.network
            val_mse, val_mae, val_mape, model = self.train_class.train_regression(trial_network, trial_config)
            self._booster = model
            return val_mse
    
    def unique_suggest_float(self, trial):
        while True:
            lr = trial.suggest_categorical('lr', self.lr_values)
            if lr not in self.suggested_values:
                self.suggested_values.add(lr)
                # self.lr_values.remove(lr)
                return lr
    
    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_booster = self._booster

    def test_results(self, plot = False):
        testloader = self.train_class.testloader
        
        if self.model_type != 'classification':
            yscaler = self.train_class.y_scaler
            mse_loss, mae, mape = self.train_class.evaluate_regression_model(self.best_booster, testloader, yscaler)
            self.test_mae = mae
            self.test_mape = mape
            self.test_mse_loss = mse_loss
            
            if plot:
                self.train_class.plot_test_output_predictions(self.best_booster, self.config)
            return self.test_mse_loss, self.test_mae, self.test_mape
        else:
            loss, accuracy = self.train_class.evaluate_classification_model(self.best_booster, testloader)
            return loss, accuracy
    
    def save_best_model(self, path):
        torch.save(self.best_booster, path)
