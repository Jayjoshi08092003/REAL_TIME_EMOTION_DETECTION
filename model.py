# model.py - MiniXception for Grayscale FER2013
import torch
import torch.nn as nn

class MiniXception(nn.Module):
    def __init__(self, num_classes):
        super(MiniXception, self).__init__()

        def conv_dw(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_dw(1, 8),
            conv_dw(8, 16),
            nn.MaxPool2d(2),
            conv_dw(16, 32),
            nn.MaxPool2d(2),
            conv_dw(32, 64),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(num_classes):
    return MiniXception(num_classes)
