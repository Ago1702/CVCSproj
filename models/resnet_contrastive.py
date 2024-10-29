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
from models import loss
from utils.modules import MultiCBAM

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"Memoria allocata GPU: {allocated:.2f} MB")
    print(f"Memoria riservata GPU: {reserved:.2f} MB")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Ocio non ci sta CUDA")
    torch.backends.cudnn.enabled=False
    
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, dataset, batch_size=10,dataset_mode=DirectoryRandomDataset.COUP,num_workers=2,pacman=True)
    res_net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    res_net.fc = nn.Identity()
    res_net.avgpool = nn.Sequential(
        MultiCBAM(2048, 8),
        MultiCBAM(2048, 16),
        MultiCBAM(2048, 32),)
    res_net = res_net.cuda() 

    criterion = loss.ContrastiveLoss_V1(couple_boost=1.1)
    optimizer = optim.Adam(res_net.parameters(), lr = 0.001)
    optimizer.zero_grad()

    res_net.train()
    for params in res_net.parameters():
        params.requires_grad = False

    for params in res_net.layer4.parameters():
        params.requires_grad = True
    for params in res_net.layer3.parameters():
        params.requires_grad = True

    for params in res_net.avgpool.parameters():
        params.requires_grad = True

    running_loss = 0

    res_net = nn.DataParallel(res_net)  
    if path.exists('/work/cvcs2024/VisionWise/weights/res_weight_contrastive.pth'):
        print("Loaded from file", flush=True)
        res_net.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/res_weight_contrastive.pth', weights_only=True))
        res_net.eval()
    


    optimizer.zero_grad()

    print("Start learning", flush=True)
    for n, (images, labels) in enumerate(dataloader):
        #print(n, flush=True)

        out = res_net(images)

        #print(out.shape, labels.shape)
        loss = criterion(out, labels.float())

        loss.backward()
        if((n + 1) % 10 == 0):  
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        if((n + 1) % 50 == 0):
            torch.save(res_net.state_dict(), '/work/cvcs2024/VisionWise/weights/res_weight_contrastive.pth')
            print(f"Iteration {n + 1}, model saved! Current iteration loss is {loss.item():.4f}", flush=True)
            #print_gpu_memory_usage()
        if((n + 1) % 32000 == 0):
            print(f"Loss: {running_loss/32000:.4f}", flush=True)
            running_loss = 0
