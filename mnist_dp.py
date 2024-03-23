import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_resnet():
    model = models.resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train():
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128 * torch.cuda.device_count(), shuffle=True)
    
    model = get_resnet()
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # Wrap the model for DataParallel
        model = nn.DataParallel(model)

    model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(5):  # loop over the dataset multiple times
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

if __name__ == "__main__":
    train()
