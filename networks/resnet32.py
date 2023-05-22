import torch
import torch.nn as nn

__all__ = ['resnet32']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, stride), padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, stride)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        # make x match the output

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=(1, 1), stride=(1, stride)),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.shortcut(x) + self.residual_function(x))

'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

'''


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.c1 = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU()
        )
        self.c2 = self._make_layer(block, self.inplanes, layers[0], stride=2)
        self.c3 = self._make_layer(block, self.inplanes, layers[1], stride=2)

        self.c4 = self._make_layer(block, self.inplanes, layers[2], stride=2)
        self.c5 = self._make_layer(block, self.inplanes, layers[3], stride=2)

        self.c6 = self._make_layer(block, self.inplanes, layers[3], stride=2)
        self.c7 = self._make_layer(block, self.inplanes, layers[3], stride=2)
        self.c8 = self._make_layer(block, self.inplanes, layers[3], stride=2)
        self.c9 = self._make_layer(block, self.inplanes, layers[3], stride=2)

        self.aver_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.linear = nn.Sequential(
            nn.Linear(self.inplanes * block.expansion * 2 * 9, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.Dropout(0.3)
        )
        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(1024, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 1)
        # torch.size(n, 1, 2, 4800)
        x = self.c1(x)

        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.aver_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.fc(x)
        return x


def resnet32(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    # change n=3 for ResNet-20, and n=9 for ResNet-56
    n = 2
    model = ResNet(BasicBlock, [n, n, n, n], **kwargs)
    return model
