import torch
import torch.nn as nn
class HW2Q3Conv2dEmbed(nn.Module):
    def __init__(self):
        super(HW2Q3Conv2dEmbed,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )
        self.final_pool = nn.AdaptiveAvgPool2d(1)

        self.linear_1 = nn.Sequential(
            nn.Linear(1024,32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(753,32),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.linear_3 = nn.Linear(64,8)
    
    def forward(self,x,y):
        x = x.unsqueeze(dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.final_pool(x)
        x = self.linear_1(x.view(x.shape[0],-1))
        y = self.linear_2(y)
        final = torch.cat((x,y),1) 
        final = self.linear_3(final)
        return final