from torch import nn
import torch


class PointPredictor(nn.Module):
    def __init__(self, n_output_boxes: int):
        self.n_output_boxes = n_output_boxes
        super(PointPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(
            128, n_output_boxes * 2
        )  # n_output_boxes bounding boxes * 2 coordinates each
        self.elu = nn.ELU()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv5(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.elu(x) + 1  # All values should be positive at the end.
        x = x.view(
            -1, self.n_output_boxes, 2
        )  # reshape to (batch_size, n_output_boxes, 2)
        return x
