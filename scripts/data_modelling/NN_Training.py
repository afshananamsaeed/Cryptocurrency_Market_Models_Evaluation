import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.regression import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
from data_modelling.EarlyStopping import EarlyStopping
torch.manual_seed(1)
from utilities.visualisation import plot_regression_network_output


# instantiate the correct device
device = torch.device("cuda")

class TrainNNModel:
    def __init__(self, X, y, batch_size = 64):
        self.model_type = None
        self.X = X
        self.y = y
        self.X_test_df = None
        self.y_test_df = None
        self.batchsize = batch_size
        self.trainloader = None
        self.validationloader = None
        self.testloader = None
        self.x_scaler = MinMaxScaler(feature_range = (0, 1))
        self.y_scaler = MinMaxScaler(feature_range = (0, 1))
    
    def run_training_regression(self, network, config):
        self.make_dataloader()
        val_mse_loss, val_mae, val_mape, model = self.train_regression(network, config)
        return val_mse_loss, val_mae, val_mape, model

    def run_train_regression_with_plotting(self, config, save_path):
        val_mse_loss, mae, mape = self.run_training_regression(config)
        # self.plot_regression_train_losses(val_mse_loss, mae, mape, r2, save_path)

    def get_test_train_data(self):
        X_train, X_remain, y_train, y_remain = train_test_split(self.X, self.y, train_size=0.80, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, train_size=0.5, shuffle=False)
        self.X_test_df, self.y_test_df = X_test, y_test
        return X_train, X_val, X_test, y_train, y_val, y_test

    def standardize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        X_train = torch.tensor(self.x_scaler.fit_transform(X_train), dtype = torch.float32)
        X_val = torch.tensor(self.x_scaler.transform(X_val), dtype = torch.float32)
        X_test = torch.tensor(self.x_scaler.transform(X_test), dtype = torch.float32)
        
        y_train = torch.tensor(self.y_scaler.fit_transform(y_train), dtype = torch.float32)
        y_val = torch.tensor(self.y_scaler.transform(y_val), dtype = torch.float32)
        y_test = torch.tensor(self.y_scaler.transform(y_test), dtype = torch.float32)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def make_dataloader(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.get_test_train_data()
        # normalise the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.standardize_data(X_train, X_val, X_test, y_train, y_val, y_test)

        self.trainloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=False, batch_size=self.batchsize)
        self.validationloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=False, batch_size=self.batchsize)
        self.testloader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=1)

    def train_regression(self, network_raw, config, epochs=100):
        self.make_dataloader()
        self.model_type = network_raw.__class__.__name__
        # initialise the network
        network = network_raw(config)
        network = network.to(device)
        
        loss = torch.nn.MSELoss()
        
        lr = config['eta']

        optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

        early_stopping = EarlyStopping(tolerance_early_stop = 5, tolerance_training_rate = 10, min_delta = 10)
        
        # collect loss values and accuracies over the training epochs
        train_loss, val_mse_loss, val_mae, val_mape = [], [], [], []

        for epoch in range(epochs):
            # train network on training data
            for x,t in self.trainloader:
                optimizer.zero_grad()
                x = x.view(x.shape[0], 1, x.shape[1])
                x = x.to(device)
                t = t.to(device)
                z = network(x)
                J = loss(z, t)
                train_loss.append(J)
                J.backward()
                optimizer.step()

            # test network on test data
            with torch.no_grad():
                val_mse_loss_epoch, val_mae_epoch, val_mape_epoch = TrainNNModel.evaluate_regression_model(network, self.validationloader, self.y_scaler)
                
                val_mse_loss.append(val_mse_loss_epoch)
                val_mae.append(val_mae_epoch)
                val_mape.append(val_mape_epoch)

            scheduler.step(val_mse_loss_epoch)
                
            # check for early stopping
            if early_stopping.early_stop_check(val_mse_loss[-1]):
                break
            
        # return the final validation loss
        return val_mse_loss_epoch, val_mae_epoch, val_mape_epoch, network
    
    def train_classification(self, network, config, epochs=100):
        
        self.make_dataloader()
        self.model_type = network.__class__.__name__
        network = network(config, classification = True)
        network = network.to(device)
        
        loss = torch.nn.CrossEntropyLoss()
    
        lr = config['eta']
        
        optimizer = torch.optim.Adam(params=self.network.parameters(), lr=lr)

        early_stopping = EarlyStopping(tolerance_early_stop = 5, tolerance_training_rate = 3, min_delta = 10)
        
        # instantiate the correct device
        device = torch.device("cuda")
        network = self.network.to(device)

        # collect loss values and accuracies over the training epochs
        train_loss, val_loss, val_acc = [], [], []

        for epoch in range(epochs):
            # train network on training data
            for x,t in self.trainloader:
                optimizer.zero_grad()
                x = x.view(x.shape[0], 1, x.shape[1])
                x = x.to(device)
                t = t.to(device)
                z = network(x)
                J = loss(z, t)
                train_loss.append(J)
                J.backward()
                optimizer.step()

            # test network on test data
            with torch.no_grad():
                correct = 0
                test_loss = []
                predicted_class = []
                actual_class = []
                
                for x,t in self.testloader:
                    x = x.view(x.shape[0], 1, x.shape[1])
                    z = network(x.to(device))
                    J = loss(z, t.to(device))
                    test_loss.append(J.item())
                    correct += torch.sum(torch.argmax(z, dim=1) == t.to(device)).item()
                    predicted_class.extend(torch.argmax(z, dim=1).tolist())
                    actual_class.extend(t)
                
                val_loss.append(sum(test_loss) / len(test_loss))   
                acc = correct / len(self.y_test)
                val_acc.append(acc)
            
            # check for decrease training rate
            # if early_stopping.decrease_training_rate(train_loss[-1]):
            #     optimizer = torch.optim.Adam(params=self.network.parameters(), lr=lr/2) 

            # check for early stopping
            if early_stopping.early_stop_check(val_loss[-1]):
                break
            
        with torch.no_grad():
            test_mse, test_mae, test_mape = TrainNNModel.evaluate_regression_model(network, self.testloader, self.y_scaler)

        return test_mae
    
    @staticmethod
    def rolling_average(values, window_size):
        return np.convolve(values, np.ones(window_size) / window_size, mode='valid')
    
    @staticmethod
    def evaluate_regression_model(network, testloader, y_scaler):
        y_actual = []
        y_pred = []
        loss = torch.nn.MSELoss()
        mse_loss_epoch = []
        mape_loss_epoch = []
        mae_loss_epoch = []
        with torch.no_grad():
            for x,t in testloader:
                x = x.view(x.shape[0], 1, x.shape[1])
                z = network(x.to(device))
                
                # use inverse scaler - ######
                t = y_scaler.inverse_transform(t)
                z = y_scaler.inverse_transform(z.cpu().detach().numpy())
                
                y_actual.extend(t)
                y_pred.extend(z)
                
                t = torch.tensor(t, dtype = torch.float32, device=device)
                z = torch.tensor(z, dtype = torch.float32, device=device)
                #############################
                
                J = loss(z, t.to(device))
                mape_criterion = MeanAbsolutePercentageError().to(device)
                mae_criterion = MeanAbsoluteError().to(device)
                
                mse_loss_epoch.append(J.item())
                mape_loss_epoch.append(mape_criterion(z, t.to(device)).item())
                mae_loss_epoch.append(mae_criterion(z, t.to(device)).item())
                
            mse_loss = sum(mse_loss_epoch) / len(mse_loss_epoch)
            mae = sum(mae_loss_epoch) / len(mae_loss_epoch)
            mape = sum(mape_loss_epoch) / len(mape_loss_epoch)
            
            return mse_loss, mae, mape
        
    @staticmethod
    def evaluate_classification_model(network, testloader):
        loss = torch.nn.MSELoss()
        correct = 0
        test_loss = []
        predicted_class = []
        actual_class = []
        
        for x,t in testloader:
            x = x.view(x.shape[0], 1, x.shape[1])
            z = network(x.to(device))
            J = loss(z, t.to(device))
            test_loss.append(J.item())
            correct += torch.sum(torch.argmax(z, dim=1) == t.to(device)).item()
            predicted_class.extend(torch.argmax(z, dim=1).tolist())
            actual_class.extend(t)
        
        test_loss = sum(test_loss) / len(test_loss)
        test_acc = correct / len(actual_class)
        return test_loss, test_acc  
    
    def plot_test_output_predictions(self, network, config):
        
        forecasting = False
        forecasting_period = 0
        if config['output_dimension']>1:
            forecasting = True
            forecasting_period = config['output_dimension']
            
        metric = config['metric']
        label = config['label']
        processed = config['processed']
        sentiment_past = config['sentiment_past']
        price_past = config['price_past']
        model_type = config['model_type']

        plt = plot_regression_network_output(network, self.X_test_df, self.y_test_df, self.x_scaler, self.y_scaler, forecasting = forecasting, forecasting_period=forecasting_period)
        path = '/home/ubuntu/Masters_Thesis/results/prediction_plots/'
        filename = f'{network.__class__.__name__}_{model_type}_{metric}_{label}_processed_{processed}_SPF_{sentiment_past}_{price_past}_{forecasting_period}.pdf'
        fig_path = path+filename
        plt.savefig(fig_path, bbox_inches='tight', facecolor='white')