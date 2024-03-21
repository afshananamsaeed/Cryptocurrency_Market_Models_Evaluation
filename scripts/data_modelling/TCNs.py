import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import numpy as np

TCN_config = {
    'input_dimension': [],
    'kernel_size': 2,
    'layers': 2,
    'output_dimension': [],
    'num_channels': 1,
    'dropout': 0.1,
    'eta': 0.0001
}

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(num_channels):
            dilation_size = 2 ** i
            in_channels = num_channels
            out_channels = 64
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
class TCN(nn.Module):
    def __init__(self, config, classification = False):
        super(TCN, self).__init__()
        self.input_size = config['input_dimension']
        self.num_channels = config['num_channels'] 
        self.kernel_size = config['kernel_size']
        self.dropout = config['dropout']
        self.output_size = config['output_dimension']
        self.classification = classification
        
        self.tcn = TemporalConvNet(self.input_size, self.num_channels, self.kernel_size, self.dropout)
        self.Flatten = torch.nn.Flatten()
        
        self.fc1 = torch.nn.Linear(64*self.input_size, 64)
        self.fc2 = torch.nn.Linear(64, self.output_size)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        x = self.tcn(x)
        x = self.Flatten(x)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(self.relu(x))
        if self.classification:
            x = self.sigmoid(x)
        return x