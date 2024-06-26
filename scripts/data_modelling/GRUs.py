import torch
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

GRU_config = {
    'input_dimension': [],
    'hidden_dimension': 512,
    'layers': 2,
    'output_dimension': [],
    'dropout': 0.1
}

class GRU(torch.nn.Module):
    def __init__(self, config, classification = False):
        super(GRU, self).__init__()
        self.input_dimension = config['input_dimension']
        self.output_dimension = config['output_dimension'] 
        self.dropout = config['dropout']
        self.hidden_dim = config['hidden_dimension']
        self.layer_dim = config['layers']
        self.classification = classification
        
        self.gru = torch.nn.GRU(self.input_dimension, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout)
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dimension)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc1(self.relu(out[:, -1, :]))
        out = self.fc2(self.relu(out))
        if self.classification:
            out = self.sigmoid(out)
        return out
