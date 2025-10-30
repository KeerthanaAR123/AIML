"""Corrected CIFAR-100 CNN example (PyTorch).

This file fixes import typos, DataLoader names, and the flatten size.
It includes a small smoke test that runs a forward pass on a dummy batch
so you can verify the model loads and the shapes are correct without
downloading the dataset.

To run the smoke test (from project root, with your venv activated):

    python cifar100_cnn_pytorch.py

If you want to enable the real dataset code, call `make_dataloaders()`
in the `__main__` section (it is provided but commented out to avoid
automatic downloading during the quick test).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_dataloaders(root='./data', batch_size=128, num_workers=2):
    """Create CIFAR-100 train/test dataloaders.

    Note: downloads the dataset if not present. Disabled by default in the
    smoke test below to keep a fast local check.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # CIFAR-100 mean/std (optional, but recommended)
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_data = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


class CIFAR100CNN(nn.Module):
    def __init__(self):
        super(CIFAR100CNN, self).__init__()
        # conv/pool sequence; with input 32x32 the final feature map after
        # three conv+pool steps (kernel sizes below) will be 2x2 (see notes).
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)   # -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # -> 12x12
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # -> 4x4
        self.pool = nn.MaxPool2d(2, 2)                 # halves spatial dims

        # After three poolings the spatial dims are 2x2 -> flattened features 128*2*2
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 100)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # robust flattening
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Quick smoke test: do a forward pass with a dummy batch to verify shapes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR100CNN().to(device)

    # Dummy input: batch size 4, 3 channels, 32x32 (CIFAR image size)
    dummy = torch.randn(4, 3, 32, 32, device=device)
    out = model(dummy)
    print("Smoke test output shape:", out.shape)  # expected (4, 100)

    # If you'd like to run with the real dataset, uncomment the lines below:
    # train_loader, test_loader = make_dataloaders()
    # print('Train batches:', len(train_loader), 'Test batches:', len(test_loader))
