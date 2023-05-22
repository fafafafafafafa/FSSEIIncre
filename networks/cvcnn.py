import torch
import torch.nn as nn

from complexPyTorch.complexLayers import *
from complexPyTorch.complexFunctions import *


class base_model_CV(nn.Module):
    def __init__(self, num_classes=10):
        super(base_model_CV, self).__init__()

        self.c1 = nn.Sequential(
            ComplexConv2d(1, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # [b,64,1,2400]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,1200]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,600]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,300]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,150]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,75]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,37]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,18]

            ComplexConv2d(64, 64, kernel_size=(1, 3), padding='same'),
            ComplexBatchNorm2d(64),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=(1, 2)),  # # [b,64,1,9]
        )

        self.dense = ComplexLinear(64 * 9, 1024)
        self.fc = Linear(1024, num_classes)
        self.head_var = 'fc'

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        # torch.size(n, 2, 4800)
        x1 = x[:, 0, :]     # torch.size(n, 4800)
        x2 = x[:, 1, :]
        x = x1 + 1j*x2  # torch.size(n, 4800)
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 1)
        # torch.size(n, 1, 1, 4800)
        x = self.c1(x)
        x = x.view(-1, 64 * 9)
        x = self.dense(x)
        x = x.abs()

        x = self.fc(x)
        return x


def cv_cnn(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError

    model = base_model_CV()
    return model
