import torch
import torch.nn as nn
import os
import sys
from utils.modules import MultiCBAM
from iter_dataset import DirectoryRandomDataset
from utils.transform import RandomTransform
import torch.optim as optim
from dataloader 

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
    dataloader = 
    model = DummyCBAM_1().cuda()
    iterator = dataset.__iter__()

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size=32,  # Number of samples per batch
    shuffle=True,  # Shuffle the dataset at the beginning of each epoch
    num_workers=4,  # Use multiple subprocesses for data loading
    transform = RandomTransform(p=0.8,scale=1,cropping_mode=RandomTransform.GLOBAL_CROP,pacman=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        
        images = []
        labels = []
        for _ in range(32):
            try:
                next = iterator.__next__()
                image = next[0].to(torch.float32).to(torch.device("cuda"))
                image = transform.forward(image)
                image = torch.squeeze(image,axis=0)
                label = next[1].to(torch.device('cuda')).unsqueeze(0).to(torch.float32)
                images.append(image)  # Append to the list
                labels.append(label)
            except StopIteration:
                break  # Break if there are no more images

        stacked_images = torch.stack(images)
        stacked_labels = torch.stack(labels)


        model.train()
 
        optimizer.zero_grad()
        
        # 5.2. Forward pass
        outputs = model(stacked_images)
        
        # 5.3. Calcolo della perdita
        loss = criterion(outputs, stacked_labels)
        
        # 5.4. Backward pass e ottimizzazione
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    dataset2 = DirectoryRandomDataset(max_iter = 5000, dir='/work/cvcs2024/VisionWise/test')
    iterator2 = dataset.__iter__()
    images = []
    labels = []
    for _ in range(100):
        try:
            next = iterator2.__next__()
            image = next[0].to(torch.float32).to(torch.device("cuda"))
            image = transform.forward(image)
            image = torch.squeeze(image,axis=0)
            label = next[1].to(torch.device('cuda')).unsqueeze(0).to(torch.float32)
            images.append(image)  # Append to the list
            labels.append(label)
        except StopIteration:
            break  # Break if there are no more images

    stacked_images = torch.stack(images)
    stacked_labels = torch.stack(labels)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(stacked_images)  # X_test should be a PyTorch tensor
        predictions = (outputs > 0.5).float()
        correct += (predictions == stacked_labels).sum().item()
        total += stacked_labels.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy}')