#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        # Input layer with batch normalization
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.bn_input = nn.BatchNorm1d(dim_hidden)
        
        # Dropout layer with a defined dropout rate
        self.dropout = nn.Dropout(dropout_rate)
        
        # Hidden layer with batch normalization
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.bn_hidden = nn.BatchNorm1d(dim_out)
        
    def forward(self, x):
        # Flatten input if it's not already flattened
        x = x.view(x.size(0), -1)
        
        # First layer with batch norm, dropout, and activation
        x = self.layer_input(x)
        x = self.bn_input(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        
        # Output layer with batch norm (optional) and LogSoftmax for classification
        x = self.layer_hidden(x)
        x = self.bn_hidden(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        
        # Convolutional layers with increased filters and smaller kernels
        self.conv1 = nn.Conv2d(args.num_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Dropout layer
        self.dropout = nn.Dropout2d(p=0.3)
        
        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Adjust based on your input
        
        # Fully connected layers with adjusted sizes
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # Adjusted based on adaptive pooling size
        self.fc2 = nn.Linear(128, args.num_classes)
    
    def forward(self, x):
        # Convolutional layer 1 with ReLU, batch norm, and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Convolutional layer 2 with ReLU, batch norm, dropout, and pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(self.dropout(x), 2)
        
        # Adaptive pooling to fix output size before flattening
        x = self.adaptive_pool(x)
        
        # Flatten layer
        x = x.view(-1, 32 * 4 * 4)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer with log softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNPathMnist(nn.Module):
    def __init__(self, args):
        super(CNNPathMnist, self).__init__()
        
        # Convolutional layer 1 with more filters and batch normalization
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Convolutional layer 2 with more filters and batch normalization
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Convolutional layer 3 to further deepen the network
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(3 * 3 * 128, 128)
        self.fc2 = nn.Linear(128, args.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Flattening layer for the fully connected layers
        out = out.view(out.size(0), -1)
        
        # Fully connected layers with dropout and ReLU
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        
        # Final output layer with LogSoftmax for stability
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        
        # First block: Convolutional layer with more filters and batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 16x16
            nn.Dropout2d(0.25)
        )
        
        # Second block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 8x8
            nn.Dropout2d(0.25)
        )
        
        # Third block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces to 4x4
            nn.Dropout2d(0.25)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        
    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Forward pass through fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # LogSoftmax for stability
        return F.log_softmax(x, dim=1)

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10):
        super(AllConvNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.01)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.01)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.01)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1, stride=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.01)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.01)
        )

        self.class_conv = nn.Conv2d(192, n_classes, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = self.class_conv(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return x
