import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pooling import DiffStride1d

"""
https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
"""

class M5(nn.Module):
    def __init__(
        self, 
        n_input: int = 1, 
        n_output: int = 35, 
        stride: int = 16, 
        n_channel: int = 32, 
        downsample_type: str = "default"
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        if downsample_type == "default":
            self.pool1 = nn.MaxPool1d(4)
            self.pool2 = nn.MaxPool1d(4)
            self.pool3 = nn.MaxPool1d(4)
            self.pool4 = nn.MaxPool1d(4)
        elif downsample_type == "diffstride":
            self.pool1 = DiffStride1d(stride=4.0)
            self.pool2 = DiffStride1d(stride=4.0)
            self.pool3 = DiffStride1d(stride=4.0)
            self.pool4 = DiffStride1d(stride=4.0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
