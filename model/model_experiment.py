import torch
import torch.nn as nn
import os
import sys
from utils.modules import MultiCBAM
from iter_dataset import DirectoryRandomDataset
from utils.transform import RandomTransform
import torch.optim as optim
from dataloader import TransformDataLoader

class DummyCBAM_1(nn.Module):
    def __init__(self):
        super(DummyCBAM_1, self).__init__()
        # defining layers
        self.net = nn.Sequential(
        nn.Conv2d(3, 9, 3, padding=2),
        nn.Conv2d(9, 27, 4, padding=3),
        nn.Conv2d(27, 81, 5, padding=4),
        MultiCBAM(81, 8),
        MultiCBAM(81, 16),
        MultiCBAM(81, 32),
        nn.Conv2d(81, 27, 5),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(27, 9, 4),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(9, 3, 3),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Conv2d(3, 1, 3),
        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Flatten(),
        nn.Linear(100,1),
        nn.Sigmoid()
    )
        
    def forward(self,x:torch.Tensor):
        return self.net.forward(x)
    
if __name__ == '__main__':
    # checking if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Did you use the slurm force?")
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/test')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP,dataset,num_workers=1)
    model = DummyCBAM_1().cuda()

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for images, labels in dataloader:
        
        #model in training moden
        model.train()

        #zeroing the gradient in the optimizer
        optimizer.zero_grad()
        
        #forward pass
        outputs = model(images)
        
        #calcolo della loss
        loss = criterion(outputs, labels)
        
        #backprop e ottimizzazione
        loss.backward()
        optimizer.step()
        print('Loss: {loss.item():.4f}')