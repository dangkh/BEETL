import torch
import math

import torch.nn as nn
import torch.nn.functional as F


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
        self.drop1 = nn.Dropout(0.25)
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

class LSTMNet_t1_65(nn.Module):
    def __init__(self, n_classes):
        super(LSTMNet_t1_65, self).__init__()
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


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        else:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WvConvNet(nn.Module):
    def __init__(self, n_classes, depth, widen_factor=1, drop_rate=0.5, flatten=True, stride=1):
        super(WvConvNet, self).__init__()
        self.drop_fc = nn.Dropout(0.1)
        self.drop_conv = nn.Dropout2d(0.1)
        self.n_classes = n_classes
        self.conv_f_size = 32

        self.conv_fs = nn.ModuleList()
        self.bn_fs = nn.ModuleList()
        self.pool_fs = nn.ModuleList()
        for i in range(4):
            self.conv_fs.append(nn.Conv2d(in_channels=17, out_channels=self.conv_f_size, kernel_size=[5, 1], stride=1,
                                          dilation=(2 * (i + 1), 1)))
            self.bn_fs.append(nn.BatchNorm2d(self.conv_f_size))
            self.pool_fs.append(nn.AdaptiveAvgPool2d((32, 1)))

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(4, n_channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, stride, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = n_channels[3]

        self.linear = nn.Linear(640, n_classes)

        if flatten:
            self.final_feat_dim = 640

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        for conv in self.conv_fs:
            torch.nn.init.kaiming_normal_(conv.weight)

    def forward(self, x):
        outputs = []
        for i in range(len(self.conv_fs)):
            output = F.relu(self.conv_fs[i](x))
            output = self.bn_fs[i](self.pool_fs[i](output))
            #             output = self.drop_conv(output)
            output = torch.squeeze(output)
            outputs.append(output)

        output = torch.stack((outputs), dim=1)
        #print(output.shape)
        #         output = self.drop_conv(output)

        out = self.conv1(output)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(x.size(0), -1)

        logits = F.relu(self.linear(self.drop_fc(out)))

        return logits


import torch.nn as nn
import numpy as np


class CNN2D(torch.nn.Module):
    def __init__(self, input_size, kernel_size, conv_channels,
                 dense_size, dropout):
        super(CNN2D, self).__init__()
        self.cconv = []
        self.MaxPool = nn.MaxPool2d((1, 4), (1, 4))
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)
        self.batchnorm = []
        self.dropouts = []
        # ############ batchnorm ###########
        for jj in conv_channels:
            self.batchnorm.append(nn.BatchNorm2d(jj, eps=0.001, momentum=0.01,
                                                 affine=True, track_running_stats=True).cuda())

            self.dropouts.append(nn.Dropout(0.5))

        ii = 0  ##### define CONV layer architecture: #####
        for in_channels, out_channels in zip(conv_channels, conv_channels[1:]):
            conv_i = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size[ii],  # stride = (1, 2),
                                     padding=(kernel_size[ii][0] // 2, kernel_size[ii][1] // 2))
            self.cconv.append(conv_i)
            self.add_module('CNN_K{}_O{}'.format(kernel_size[ii], out_channels), conv_i)
            ii += 1

        #self.fc_conv = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 4), padding=(1, 2))
        #self.fc_pool = nn.MaxPool2d((3, 4))
        #self.fc_drop = nn.Dropout(0.5)
        #self.fc_conv1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 4), padding=(1, 2))
        #self.fc_pool1 = nn.MaxPool2d((3, 4))
        #self.fc_drop1 = nn.Dropout(0.5)
        self.flat_dim = self.get_output_dim(input_size, self.cconv)
        #self.flat_dim = 3040
        self.fc1 = torch.nn.Linear(self.flat_dim, dense_size)
        self.fc2 = torch.nn.Linear(dense_size, 3)

    def get_output_dim(self, input_size, cconv):
        with torch.no_grad():
            input = torch.ones(1, *input_size)
            for conv_i in cconv:
                input = self.MaxPool(conv_i(input))
                flatout = int(np.prod(input.size()[1:]))
                print("Input shape : {} and flattened : {}".format(input.shape, flatout))
        return flatout

    def forward(self, x):
        for jj, conv_i in enumerate(self.cconv):
            x = conv_i(x)
            x = self.batchnorm[jj + 1](x)
            x = self.ReLU(x)
            x = self.dropouts[jj](x)
            x = self.MaxPool(x)
            
        # flatten the CNN output
        #x = F.relu(self.fc_conv(x))
        #x = self.fc_pool(self.fc_drop(x))
        #print(x.shape)
        out = x.view(-1, self.flat_dim)
        out = F.relu(self.fc1(out))
        out = self.Dropout(out)
        out = self.fc2(out)
        return out
