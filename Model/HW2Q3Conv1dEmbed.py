import torch
import torch.nn as nn

class HW2Q3Conv1dEmbed(nn.Module):
    def __init__(self, num_mels, num_genres):
        super(HW2Q3Conv1dEmbed,self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(num_mels, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=7)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.linear_1 = nn.Sequential(
            nn.Linear(128,32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(753,32),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.linear_3 = nn.Linear(64,8)

    def forward(self,x,y):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = self.linear_1(x.squeeze(-1))
        y = self.linear_2(y)
        final = torch.cat((x,y),1) 
        final = self.linear_3(final)
        return final