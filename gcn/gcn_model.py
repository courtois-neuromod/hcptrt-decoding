import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, edge_index, edge_weight, n_timepoints, resolution, n_classes=21):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_timepoints = n_timepoints
        self.resolution = resolution
        
        # 1 input image channel, 32 output channels, 2 Chebyshev filter size
        self.conv1 = tg.nn.ChebConv(in_channels=self.n_timepoints,out_channels=32,K=2,bias=True)
        self.conv2 = tg.nn.ChebConv(in_channels=32,out_channels=32,K=2,bias=True)
        self.conv3 = tg.nn.ChebConv(in_channels=32,out_channels=32,K=2,bias=True)
        self.conv4 = tg.nn.ChebConv(in_channels=32,out_channels=32,K=2,bias=True)
        self.conv5 = tg.nn.ChebConv(in_channels=32,out_channels=32,K=2,bias=True)
        self.conv6 = tg.nn.ChebConv(in_channels=32,out_channels=32,K=2,bias=True)
        
        # affine operation: y = Wx + b
#         self.fc1 = nn.Linear(256*32, 256) #444
        self.fc1 = nn.Linear(self.resolution*32, 256) #444
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv2(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv3(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv4(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv5(x, self.edge_index, self.edge_weight))
        x = self.conv6(x, self.edge_index, self.edge_weight)

        x = tg.nn.global_mean_pool(x,torch.from_numpy(np.array(range(x.size(0)),dtype=int)))        
#         x = x.view(-1,256*32) #444
        x = x.view(-1,self.resolution*32)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
                
        return x