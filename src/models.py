import torch.nn as nn


class LeNet5(nn.Module):

  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.relu = nn.ReLU()

    self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(5 * 5 * 16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.pool1(x)

    x = self.relu(self.conv2(x))
    x = self.pool2(x)

    x = self.flatten(x)

    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)

    return x
  

class VGGNet(nn.Module):

  def __init__(self):
    super(VGGNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1 * 1 * 512, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 10)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
  

class ResBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride=1):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.skip = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.skip = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
          nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    identity = self.skip(x)

    output = self.conv1(x)
    output = self.bn1(output)
    output = self.relu(output)

    output = self.conv2(output)
    output = self.bn2(output)

    output += identity
    output = self.relu(output)

    return output


class ResNet(nn.Module):

  def __init__(self):
    super(ResNet, self).__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer1 = self.make_layer(64, 64, 2, stride=1)
    self.layer2 = self.make_layer(64, 128, 2, stride=2)
    self.layer3 = self.make_layer(128, 256, 2, stride=2)
    self.layer4 = self.make_layer(256, 512, 2, stride=2)

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 10)
    )

  def make_layer(self, in_channels, out_channels, num_blocks, stride):
    layers = []
    layers.append(ResBlock(in_channels, out_channels, stride))
    for _ in range(1, num_blocks):
      layers.append(ResBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.initial(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.classifier(x)

    return x
