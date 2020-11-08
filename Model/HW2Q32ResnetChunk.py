import torch
from torch import nn
from .ResNet import ResNet

class HW2Q32ResnetChunk(nn.Module):
    def __init__(self,num_genres=8,leaky_relu_slope=0.01):
        super(HW2Q32ResnetChunk,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1,
            bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64,64, 3, padding=1, bias=False)
        )
        
        self.res_block1 = ResNet(input_channel=64, output_channel=128)
        self.res_block2 = ResNet(input_channel=128, output_channel=192)
        self.res_block3 = ResNet(input_channel=192, output_channel=256)

        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
        )
        self.linear = nn.Linear(256, num_genres)
    
    def forward(self,x):
        x = x.unsqueeze(dim=1)
        x = self.conv_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool_block(x)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x
