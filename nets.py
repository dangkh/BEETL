import torch.nn as nn
import torch.nn.functional as F
import torch
    
class LSTMNet(nn.Module):
    def __init__(self, n_classes):
        super(LSTMNet, self).__init__()
        self.drop1 = nn.Dropout(0.75)
        self.fc = nn.Linear(1024, n_classes)
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, [25, 2], stride=[2, 1], padding = [1, 1])
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        self.mp1 = nn.AvgPool2d([75,1], stride = [20, 1])
        self.conv2 = nn.Conv2d(32, 64, [25, 1], stride=[3, 1])
        self.bnorm = nn.BatchNorm2d(64, momentum=0.1, affine=True)
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        self.mp2 = nn.AvgPool2d([10,1], stride = [5, 1])
        self.lstm = nn.LSTM(3, 8, batch_first=True)

    def forward(self, x):
        output = self.conv1(x)
        output =  F.relu(output)
        output = self.mp1(output)
        output = self.conv2(output)
        output = self.bnorm(output)
        output =  F.relu(output)
        output = self.mp2(output)
        shape = output.shape
        preLSTM = output.view(len(output), shape[1] * shape[2], -1)
        h_lstm, _ = self.lstm(preLSTM)
        output = torch.flatten(h_lstm,1)
        output = self.drop1(output)
        scores = self.fc(output)
        return scores


class LSTMNet_t2(nn.Module):
    def __init__(self, n_classes):
        super(LSTMNet_t2, self).__init__()
        self.drop1 = nn.Dropout(0.75)
        self.fc = nn.Linear(5120, n_classes)
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, [10, 3], stride=[3, 2], padding = [1, 1])
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        self.mp1 = nn.AvgPool2d([10,1], stride = [2, 1])
        self.conv2 = nn.Conv2d(32, 64, [10, 2], stride=[2, 1])
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        self.mp2 = nn.AvgPool2d([10,1], stride = [2, 1])
        self.lstm = nn.LSTM(8, 16, batch_first=True)

    def forward(self, x):
        output = self.conv1(x)
        output =  F.relu(output)
        output = self.mp1(output)
        output = self.conv2(output)
        output =  F.relu(output)
        output = self.mp2(output)
        shape = output.shape
        preLSTM = output.view(len(output), shape[1] * shape[2], -1)
        h_lstm, _ = self.lstm(preLSTM)
        output = torch.flatten(h_lstm,1)
        output = self.drop1(output)
        scores = self.fc(output)
        # scores = self.sm(output)
        # stop
        return scores