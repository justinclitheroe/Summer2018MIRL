import torch.nn as nn
import torch.nn.functional as F




class CNNGood(nn.Module):
    """
    A standard CNN module pulled from a pytorch tutorial github about mnist

    This one is called "good" because it works as intended
    """
    def __init__(self):
        super(CNNGood, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = self.forward(x, training=False)
        return x

class CNN(nn.Module):
    """
    A gutted version of the above CNN

    Created to not be as effective as it once was
    """
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(1, 20, kernel_size=20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 10)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        # a function to predict the labels of a batch of inputs
        x = self.forward(x, training=False)
        return x