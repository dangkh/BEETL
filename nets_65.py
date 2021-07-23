import torch.nn as nn
import torch.nn.functional as F
import torch
    
class LSTMNet(nn.Module):
    def __init__(self, n_classes):
        super(LSTMNet, self).__init__()
        self.drop1 = nn.Dropout(0.75)
        self.fc = nn.Linear(2048, n_classes)
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, [10, 2], stride=[2, 1], padding = [1, 1])
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        self.mp1 = nn.AvgPool2d([10,1], stride = [2, 1])
        self.conv2 = nn.Conv2d(32, 64, [10, 1], stride=[3, 1])
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        self.mp2 = nn.AvgPool2d([10,1], stride = [3, 1])
        self.conv3 = nn.Conv2d(64, 128, [16, 1], stride=[4, 1])
        torch.nn.init.xavier_uniform_(self.conv3.weight, gain=1)
        self.avg = nn.AvgPool2d([8,1])
        self.lstm = nn.LSTM(3, 8, batch_first=True)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.conv1(x)
        output =  F.relu(output)
        output = self.mp1(output)
        output = self.conv2(output)
        output =  F.relu(output)
        output = self.mp2(output)
        output = self.conv3(output)
        output =  F.relu(output)
        output = self.avg(output)
        shape = output.shape
        preLSTM = output.view(len(output), shape[1] * shape[2], -1)
        h_lstm, _ = self.lstm(preLSTM)
        output = torch.flatten(h_lstm,1)
        output = self.drop1(output)
        scores = self.fc(output)
        # scores = self.sm(output)
        # stop
        return scores