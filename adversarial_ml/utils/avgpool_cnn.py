import torch
import torch.nn as nn
import torch.nn.functional as fu


class AveragePoolCNN(nn.Module):

    def __init__(self, channels, num_classes):
        super(AveragePoolCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(5, 5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=1)
        self.conv8 = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)

        self.do1 = nn.Dropout(0.3)
        self.do2 = nn.Dropout(0.3)
        self.do3 = nn.Dropout(0.3)
        self.do4 = nn.Dropout(0.5)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """ Performs forward pass. """
        x = self.bn1(fu.relu(self.conv1(x), inplace=False))
        x = self.bn2(fu.relu(self.conv2(x), inplace=False))
        x = self.do1(fu.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2)))

        x = self.bn3(fu.relu(self.conv3(x), inplace=False))
        x = self.bn4(fu.relu(self.conv4(x), inplace=False))
        x = self.do2(fu.avg_pool2d(x, kernel_size=(2, 3), stride=(2, 3)))

        x = self.bn5(fu.relu(self.conv5(x), inplace=False))
        x = self.bn6(fu.relu(self.conv6(x), inplace=False))
        x = self.do3(fu.avg_pool2d(x, kernel_size=(2, 3), stride=(2, 3)))

        x = self.do4(self.bn7(fu.relu(self.conv7(x), inplace=False)))

        x = self.conv8(x)
        x = torch.mean(x, dim=-2)    # frequency
        x = torch.max(x, dim=-1)[0]  # time

        return x

    @torch.no_grad()
    def probabilities(self, x):
        """ Returns logits after normalisation (softmax). """
        logits = self.forward(x)
        return self.softmax(logits)

    @torch.no_grad()
    def predict(self, x):
        """ Returns class with highest probability (i.e. prediction). """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
