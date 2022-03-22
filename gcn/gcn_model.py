import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import numpy as np

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)    
    model = model.double()   
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.double())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss, current = loss.item(), batch * dataloader.batch_size

        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= X.shape[0]
#         print(f"#{batch:>5};\ttrain_loss:{loss:>0.3f};\ttrain_accuracy:{(100*correct):>5.1f}%\t\t[{current:>5d}/{size:>5d}]")

def valid_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model.forward(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size
    correct /= size

    return loss, correct  


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
        self.fc1 = nn.Linear(self.resolution*32, 256) # tunning  256
        self.fc2 = nn.Linear(256, 128) # tunning  128
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.25) # tunning 0.25

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

            
    
class GCN_2layers_tunning(torch.nn.Module):    
    def __init__(self, edge_index, edge_weight, n_timepoints, resolution, n_channels, 
                 dropout, fc1_out, fc2_out, n_chebfilter):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_timepoints = n_timepoints
        self.resolution = resolution
        self.n_channels = n_channels
        self.dropout = dropout
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out
        self.n_chebfilter = n_chebfilter
        
        n_classes=21
        
        # 1 input image channel, 32 output channels, 2 Chebyshev filter size
        self.conv1 = tg.nn.ChebConv(in_channels=self.n_timepoints, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv2 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        # affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.resolution*n_channels, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):        
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = self.conv2(x, self.edge_index, self.edge_weight)

        x = tg.nn.global_mean_pool(x,torch.from_numpy(np.array(range(x.size(0)),dtype=int)))        

        x = x.view(-1,self.resolution*self.n_channels)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
                
        return x    
    
    
    
class GCN_6layers_tunning(torch.nn.Module):
    
    def __init__(self, edge_index, edge_weight, n_timepoints, resolution, n_channels, 
                 dropout, fc1_out, fc2_out, n_chebfilter):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_timepoints = n_timepoints
        self.resolution = resolution
        self.n_channels = n_channels
        self.dropout = dropout
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out
        self.n_chebfilter = n_chebfilter
        
        n_classes=21
        
        # 1 input image channel, 32 output channels, 2 Chebyshev filter size
        self.conv1 = tg.nn.ChebConv(in_channels=self.n_timepoints, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv2 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv3 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv4 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv5 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv6 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        # affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.resolution*n_channels, fc1_out) #444
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):        
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv2(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv3(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv4(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv5(x, self.edge_index, self.edge_weight))
        x = self.conv6(x, self.edge_index, self.edge_weight)

        x = tg.nn.global_mean_pool(x,torch.from_numpy(np.array(range(x.size(0)),dtype=int)))        

        x = x.view(-1,self.resolution*self.n_channels)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
                
        return x 

    
    
    
class GCN_4layers_tunning(torch.nn.Module):
    
    def __init__(self, edge_index, edge_weight, n_timepoints, resolution, n_channels, 
                 dropout, fc1_out, fc2_out, n_chebfilter):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_timepoints = n_timepoints
        self.resolution = resolution
        self.n_channels = n_channels
        self.dropout = dropout
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out
        self.n_chebfilter = n_chebfilter
        
        n_classes=21
        
        # 1 input image channel, 32 output channels, 2 Chebyshev filter size
        self.conv1 = tg.nn.ChebConv(in_channels=self.n_timepoints, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv2 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv3 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        self.conv4 = tg.nn.ChebConv(in_channels=n_channels, out_channels=n_channels,
                                    K=n_chebfilter, bias=True)
        
        # affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.resolution*n_channels, fc1_out) #444
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):        
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv2(x, self.edge_index, self.edge_weight))
        x = F.relu(self.conv3(x, self.edge_index, self.edge_weight))
        x = self.conv4(x, self.edge_index, self.edge_weight)

        x = tg.nn.global_mean_pool(x,torch.from_numpy(np.array(range(x.size(0)),dtype=int)))        

        x = x.view(-1,self.resolution*self.n_channels)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
                
        return x  

