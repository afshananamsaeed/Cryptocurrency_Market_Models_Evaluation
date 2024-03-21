import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

LSTM_config = {
    'input_dimension': [],
    'hidden_dimension': 512,
    'layers': 2,
    'output_dimension': [],
    'dropout': 0.1,
    'eta': []
}

class LSTM(torch.nn.Module):
    def __init__(self, config, classification = False):
        super(LSTM, self).__init__()
        self.input_dimension = config['input_dimension']
        self.dropout = config['dropout']
        self.output_dimension = config['output_dimension']
        self.hidden_dim = config['hidden_dimension']
        self.layer_dim = config['layers']
        self.classification = classification

        # LSTM model 
        self.lstm = torch.nn.LSTM(self.input_dimension, self.hidden_dim, num_layers = self.layer_dim, batch_first=True, dropout = self.dropout)
        self.fc1 = torch.nn.Linear(self.hidden_dim, 128) 
        self.fc2 = torch.nn.Linear(128, self.output_dimension)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc1(self.relu(out[:, -1, :]))
        out = self.fc2(self.relu(out))
        if self.classification:
            out = self.sigmoid(out)
        return out