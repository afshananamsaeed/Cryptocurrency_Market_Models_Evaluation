import torch
import numpy as np

CNN_config = {
    'input_dimension': [],
    'Q1': 128,
    'Q2': 512,
    'output_dimension': [],
    'kernel_size': 2,
    'dropout': 0.1,
    'eta': 0.0001
}

class CNN(torch.nn.Module):
    def __init__(self, config, classification = False):
        super(CNN, self).__init__()
        self.Q1 = config['Q1']
        self.Q2 = config['Q2']
        self.input_dimension = config['input_dimension']
        self.output_dimension = config['output_dimension']
        self.kernel_size = config['kernel_size']
        self.dropout = config['dropout']
        self.classification = classification
        self.output_size = (self.input_dimension - self.kernel_size) + 1

        self.Conv1d = torch.nn.Conv1d(in_channels=1, out_channels=self.Q1, kernel_size = self.kernel_size)
        self.Pool = torch.nn.AvgPool1d(kernel_size=2)
        self.act = torch.nn.ReLU()
        self.Flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(self.Q1*int((np.floor(self.output_size/2))), self.Q2)
        self.fc2 = torch.nn.Linear(self.Q2, self.output_dimension)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.Conv1d(x)
        x = self.Pool(x)
        x = self.act(x)
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.classification:
            logits = self.sigmoid(x)
            return logits
        else:
            return x