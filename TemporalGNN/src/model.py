"""
Temporal Graph Neural Network model for transaction classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Temporal GNN Model for Edge Classification
class TemporalEdgeClassifier(nn.Module):
    def __init__(self, config):
        super(TemporalEdgeClassifier, self).__init__()
        node_dim = config['model']['node_dim']
        edge_dim = config['model']['edge_dim']
        hidden_dim = config['model']['hidden_dim']
        dropout = config['model']['dropout']
        
        self.rnn = nn.GRUCell(node_dim, hidden_dim)
        self.gnn1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.gnn2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.gnn3 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.dropout = nn.Dropout(p=dropout)
        # self.classifier = nn.Linear(hidden_dim * 2 + edge_dim, 1)  # Binary classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, data, h):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Update node hidden states with RNN (using current x)
        h = self.rnn(x, h)
        
        # Apply GNN layers
        h = F.relu(self.gnn1(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.gnn2(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.gnn3(h, edge_index))
        h = self.dropout(h)

        # Edge features: concat sender h, receiver h, edge_attr
        h_i = h[edge_index[0]]
        h_j = h[edge_index[1]]
        edge_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        
        # Prediction
        out = self.classifier(edge_input)
        
        return out, h  # Return logits and updated h
    
def create_model(config):
    model = TemporalEdgeClassifier(config)
    return model