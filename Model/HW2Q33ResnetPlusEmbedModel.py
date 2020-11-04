import torch
from torch import nn
from .ResNet import ResNet

class HW2Q33ResnetPlusEmbedModel(nn.Module):
    def __init__(self,num_genres,leaky_relu_slope=0.01):
        super(HW2Q33ResnetPlusEmbedModel,self).__init__()
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
        self.linear_1 = nn.Sequential(torch.nn.Linear(753,64),
                                    torch.nn.ReLU())
        self.linear_2 = nn.Linear(320, num_genres)
    
    def forward(self,x,y):
        x = x.unsqueeze(dim=1)
        x = self.conv_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool_block(x)
        x = x.view(x.shape[0],-1)

        y = self.linear_1(y)
        x = torch.cat((x,y),1)
        x = self.linear_2(x)
        return x