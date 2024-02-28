import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        x = self.linear(x)
        x = torch.matmul(adj_matrix, x)
        return x

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, output_size)

    def forward(self, x, adj_matrix):
        x = F.relu(self.gcn1(x, adj_matrix))
        x = self.gcn2(x, adj_matrix)
        return x

class GraphLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GraphLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,dropout=0.2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # Extracting hidden state from the last layer

class GraphLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, num_layers):
        super(GraphLSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, x, h_n):
        #inverse the order of x sequence,drop the last sequence and add a zero sequence at the beginning
        x = torch.flip(x, dims=[1])
        x = torch.cat([torch.zeros_like(x[:,0,:]).unsqueeze(1),x[:,:-1,:]],dim=1)

        # Expand h_n to (num_layer,batch,input_size) from (batch,input_size) apart from the first layer, other layers are 0
        h_n_expanded = torch.cat([h_n.unsqueeze(0)] + [torch.zeros_like(h_n).unsqueeze(0)] * (self.num_layers - 1), dim=0)
        output, _ = self.lstm(x, (h_n_expanded, torch.zeros_like(h_n_expanded))) # (batch,seq_len,input_size), ((num_layer,batch,input_size),cn)
        output = self.fc(output)

        #inverse the order of output sequence
        output = torch.flip(output, dims=[1])
        return output

class GCNLSTMAutoencoder(nn.Module):
    def __init__(self, gcn_input_size, gcn_hidden_size, gcn_output_size, lstm_hidden_size, lstm_num_layers,batch_size,seq_len, num_nodes):
        super(GCNLSTMAutoencoder, self).__init__()
        self.gcn_encoder = GCN(gcn_input_size, gcn_hidden_size, gcn_output_size)
        self.gcn_decoder = GCN(gcn_output_size, gcn_hidden_size, gcn_input_size)
        self.lstm_encoder = GraphLSTMEncoder(gcn_output_size*num_nodes, lstm_hidden_size, lstm_num_layers)
        self.lstm_decoder = GraphLSTMDecoder(gcn_output_size*num_nodes, lstm_hidden_size,gcn_output_size*num_nodes, lstm_num_layers)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.gcn_output_size = gcn_output_size

    def forward(self, x, adj_matrix):
        # GCN encoder

        gcn_encoded = self.gcn_encoder(x, adj_matrix)
        
        # LSTM encoder
        lstm_encoded = self.lstm_encoder(gcn_encoded.view(self.batch_size, self.seq_len, -1))  # Adding an additional dimension for the sequence
        
        # LSTM decoder
        lstm_decoded = self.lstm_decoder(gcn_encoded.view(self.batch_size, self.seq_len, -1),lstm_encoded)

        # GCN decoder
        gcn_decoded = self.gcn_decoder(lstm_decoded.view(self.batch_size, self.seq_len, self.num_nodes, self.gcn_output_size), adj_matrix)

        return gcn_decoded

