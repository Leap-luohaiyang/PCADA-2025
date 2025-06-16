import torch.nn as nn


class SingleClassifier(nn.Module):
    def __init__(self, num_unit=288, num_classes=7):
        super(SingleClassifier, self).__init__()
        layers = [nn.Linear(num_unit, num_classes)]

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)

        return x
