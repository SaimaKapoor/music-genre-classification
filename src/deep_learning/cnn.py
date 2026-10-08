"""Baseline CNN for GTZAN log-mel spectrograms."""
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(128*4*4,256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,num_classes),
        )
    def forward(self,x):
        return self.classifier(self.features(x))
