import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        #self.input_layer = nn.Linear(178, 16) #original
        self.input_layer = nn.Linear(178, 16)
        self.output_layer = nn.Linear(16, 5)
        
        # hyper parameters
        self.drop_out = nn.Dropout(p = 0.5)
        self.batch_norm1 = nn.BatchNorm1d(178) 
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)

    def forward(self, x):
        #x = F.sigmoid(self.input_layer(x)) # original 
        x = F.relu(self.drop_out(self.input_layer(self.batch_norm1(x))))
        x = self.output_layer(x)
    
        return x


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        #original model
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels= 6, kernel_size = 5) 
        self.pool = nn.MaxPool1d(kernel_size = 2) 
        self.conv2 = nn.Conv1d(6, 16, 5) 
        self.fc1 = nn.Linear(in_features=16 * 41, out_features=128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        self.drop_out = nn.Dropout(p = 0.5)

    def forward(self, x):
        #original model
        x = self.pool(F.relu(self.drop_out(self.conv1(x))))
        x = self.pool(F.relu(self.drop_out(self.conv2(x))))
        x = x.view(-1, 16 * 41) 
        x = F.relu(self.drop_out(self.fc1(x)))
        x = F.relu(self.drop_out(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.GRU(input_size= 1, hidden_size= 16, num_layers = 1, batch_first = True)
        self.fc = nn.Linear(in_features = 16, out_features = 5) 
        
        self.rnn2 = nn.GRU(input_size= 1, hidden_size= 32, num_layers = 1, batch_first = True, dropout = 0.2)
        self.fc = nn.Linear(in_features = 32, out_features = 5) 

    def forward(self, x):
        x, _ = self.rnn2(x) 
        x = self.fc(x[:, -1, :])
        
        return x


class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        # You may use the input argument 'dim_input', which is basically the number of features
        
        self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(in_features=dim_input, out_features=32) 
        self.fc2 = nn.Linear(in_features=16, out_features=2) 
        
        self.relu = nn.ReLU()

    def forward(self, input_tuple):
        # HINT: Following two methods might be useful
        # 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn

        seqs, lengths = input_tuple
        seqs = torch.tanh(self.fc1(seqs)) 
        seqs = pack_padded_sequence(seqs, lengths.cpu(), batch_first=True)
        seqs, h = self.rnn(seqs)
        seqs, _ = pad_packed_sequence(seqs, batch_first=True) 
        seqs = self.fc2((seqs[:, -1, :]))

        return seqs