import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# -------------------------------
# Data transforms (simplified normalization)
# -------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# -------------------------------
# CNN Model Definition
# -------------------------------
class CIFAR100CNN(nn.Module):
    def __init__(self):
        super(CIFAR100CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)   # -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # -> 12x12
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # -> 4x4
        self.pool = nn.MaxPool2d(2, 2)

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
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------
# DataLoader function
# -------------------------------
def make_dataloaders(root='./data', batch_size=128, num_workers=2):
    train_data = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# -------------------------------
# Main: Smoke test
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR100CNN().to(device)

    dummy = torch.randn(4, 3, 32, 32, device=device)
    out = model(dummy)
    print("Smoke test output shape:", out.shape)  # Expected: (4, 100)

    # Uncomment to test dataloaders
    # train_loader, test_loader = make_dataloaders()
    # print("Train batches:", len(train_loader), "Test batches:", len(test_loader))
