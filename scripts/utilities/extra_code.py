import torch
from scripts.data_modelling.EarlyStopping import EarlyStopping
from torchmetrics.classification import F1Score
from ray import tune
from ray.tune import CLIReporter


def make_dataloader(self, categorical = False):
    X_train, X_val, X_test, y_train, y_val, y_test = self.get_test_train_data()
    
    # normalise the data
    X_train, X_val, X_test, y_train, y_val, y_test = self.standardize_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    if categorical:
        self.y_train = torch.squeeze(self.y_train).long()
        self.y_test = torch.squeeze(self.y_test).long()

    self.trainloader = torch.utils.data.DataLoader(list(zip(self.X_train, self.y_train)), shuffle=False, batch_size=self.batchsize)
    self.validationloader = torch.utils.data.DataLoader(list(zip(self.X_train, self.y_train)), shuffle=False, batch_size=self.batchsize)
    self.testloader = torch.utils.data.DataLoader(list(zip(self.X_test, self.y_test)), shuffle=False, batch_size=self.batchsize)

def run_training_classification(self, config):
    self.make_dataloader(categorical=True)
    val_loss, val_acc, f1_score = self.train_classification(config)
    return val_loss, val_acc, f1_score

def run_train_classification_with_plotting(self, config, save_path):
    val_loss, val_acc, f1_score = self.run_training_classification(config)
    self.plot_classification_train_loss_and_acc(val_loss, val_acc, f1_score, save_path)

def tune_model_classification(self, config):
    config['eta'] = tune.grid_search([0.1, 0.01, 0.001])
    config['optimizer'] = tune.grid_search(['SGD', 'Adam'])
    config['tune'] = True
    config['train_type'] = 'Classification'

    reporter = CLIReporter(metric_columns=["accuracy", "loss", "f1_score"])
    analysis = tune.run(self.train_classification,
                config=config,
                verbose=3,
                resources_per_trial = {'gpu': 1, 'cpu': 2 },
                progress_reporter = reporter,
                resume = 'AUTO',
                )

    # print("Best config: ", analysis.get_best_config("accuracy", "max"))
    # Get a data frame for analyzing trial results.
    df1 = analysis.dataframe()
    df1.to_csv(f"/mnt/tune_model_analysis_classification_{self.network}.csv")
    
    best_config = analysis.get_best_config("accuracy", "max")
    self.run_train_with_plotting(best_config)

def train_classification(self, config, epochs=1000):
    loss = torch.nn.CrossEntropyLoss()
    
    lr = config['eta']
    
    optimizer = torch.optim.Adam(params=self.network.parameters(), lr=lr)

    early_stopping = EarlyStopping(tolerance_early_stop = 5, tolerance_training_rate = 3, min_delta = 10)
    
    # instantiate the correct device
    device = torch.device("cuda")
    network = self.network.to(device)

    # collect loss values and accuracies over the training epochs
    train_loss, val_loss, val_acc, f1_score = [], [], [], []

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
            
            assert len(predicted_class) == len(actual_class)
            predicted_class_tensor = torch.tensor(predicted_class, dtype=torch.long)
            actual_class_tensor = torch.tensor(actual_class, dtype=torch.long)
            #f1 score on validation data
            f1 = F1Score(num_classes=config['output_dimension'], task = 'multiclass').to(device)
            f1_score_val = f1(predicted_class_tensor, actual_class_tensor)
            f1_score.append(f1_score_val.item())
        
        # check for decrease training rate
        if early_stopping.decrease_training_rate(train_loss[-1]):
            optimizer = torch.optim.Adam(params=self.network.parameters(), lr=lr/2) 

        # check for early stopping
        if early_stopping.early_stop_check(val_loss[-1]):
            break

    # if config['tune']:
    #     # returning the last epoch scores for tuning as those are supposedly the best
    #     tune.report(loss=val_loss[-1], accuracy=val_acc[-1], f1_score =f1_score[-1])
    # else:
    return val_loss, val_acc, f1_score