import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2
from data.iter_dataset import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Ocio non ci sta CUDA")
    torch.backends.cudnn.enabled=False
    
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, dataset, 1, batch_size=64)
    res_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    res_net.fc = nn.Sequential(nn.Linear(res_net.fc.in_features, 1), nn.Sigmoid())
    if path.exists('/work/cvcs2024/VisionWise/weights/res_weight.pth'):
        print("Loaded from file", flush=True)
        res_net.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/res_weight.pth', weights_only=True))
        res_net.eval()

    res_net = res_net.cuda()
    res_net = nn.DataParallel(res_net)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(res_net.parameters(), lr = 0.001)
    optimizer.zero_grad()

    res_net.train()
    for params in res_net.parameters():
        params.requires_grad = False
    for params in res_net.layer4.parameters():
        params.requires_grad = True

    tf = v2.Resize(244, 2)
    running_loss = 0
    print("Start learning", flush=True)
    for n, (images, labels) in enumerate(dataloader):
        #print(n, flush=True)
        images = tf(images)

        optimizer.zero_grad()

        out = res_net(images)

        #print(out.shape, labels.shape)
        loss = criterion(out, labels.unsqueeze(-1).float())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if((n + 1) % 50 == 0):
            torch.save(res_net.state_dict(), '/work/cvcs2024/VisionWise/weights/res_weight.pth')
            print(f"Iteration {n + 1}, model saved! Current iteration loss is {loss.item():.4f}", flush=True)
        if((n + 1) % 32000 == 0):
            print(f"Loss: {running_loss/32000:.4f}", flush=True)
            running_loss = 0
