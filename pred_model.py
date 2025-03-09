from torch import nn
import torch


class SerialPointPredictor(nn.Module):
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs
        super(SerialPointPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(576 + 2 * n_outputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # n_output_boxes bounding boxes
        self.elu = nn.ELU()

    def __repr__(self):
        return f"SerialPointPredictor_{self.n_outputs}"

    def forward(self, x):
        batch_size = x.size(0)

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
        img_encoding = x.view(x.size(0), -1)

        preds = -torch.ones(batch_size, self.n_outputs, 2)

        for i in range(self.n_outputs):
            x = torch.concat((img_encoding, preds.flatten(1)), 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = 1 + self.elu(self.fc3(x))  # All values should be positive at the end.
            preds[:, i] = x

        return preds * 100


class ParallelPointPredictor(nn.Module):
    def __init__(self, n_outputs: int):
        self.n_outputs = n_outputs
        super(ParallelPointPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2 * n_outputs)  # n_outputs bounding boxes
        self.elu = nn.ELU()

    def __repr__(self):
        return f"ParallelPointPredictor_{self.n_outputs}"

    def forward(self, x):
        batch_size = x.size(0)
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
        img_encoding = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(img_encoding))
        x = torch.relu(self.fc2(x))
        x = 1 + self.elu(self.fc3(x))  # All values should be positive at the end.
        # shape is batch_size, 2 * n_outputs
        return x.reshape(batch_size, self.n_outputs, 2) * 100
