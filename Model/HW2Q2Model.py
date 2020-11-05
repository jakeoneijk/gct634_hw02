import torch
import torch.nn as nn
class HW2Q2Model(nn.Module):
    def __init__(self,feature_size = 753 , num_genres = 8):
        super(HW2Q2Model,self).__init__()
        self.feature_size = feature_size
        self.linear = nn.Sequential(torch.nn.Linear(feature_size,32),
                                    torch.nn.BatchNorm1d(32),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(32,num_genres))
    
    def forward(self,x):
        x = self.linear(x)
        return x