import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        main = F.relu(self.bn1(self.conv1(x)))
        main = self.bn2(self.conv2(main))
        shortcut = self.shortcut(x)
        return F.relu(main + shortcut)


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name, and make sure Net() works with no arguments.
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model has 101,770 parameters, which is over the 100,000 limit.
    """
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(Net, self).__init__()

        # We define the layers of our model here by instantiating layer objects.
        # Here we define two fully connected (linear) layers.
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.block1 = ResNetBlock(24, 32)
        self.block2 = ResNetBlock(32, 48)
        self.classifier = nn.Linear(48 * 3 * 3, 10)


    def forward(self, x):
        # Define the forward pass of the model. This is where we specify how the input data flows through the layers.
        x = self.stem(x)
        x = F.max_pool2d(self.block1(x), 2)
        x = F.max_pool2d(self.block2(x), 2)
        x = F.adaptive_avg_pool2d(x, (3, 3))
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)