import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2
from data.datasets import DirectorySequentialDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
from models import loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import resnet_cbam
from torch.amp import autocast, GradScaler
from utils import notifier
from torch.utils.data import DataLoader

if __name__ == "__main__":
    iteration_index = 13000
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #uncomment this to false to debug
    torch.backends.cudnn.enabled=False

    dataset = DirectorySequentialDataset('/work/cvcs2024/VisionWise/test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    
    res_net =resnet_cbam.v4().cuda()

    classifier = nn.Sequential(nn.BatchNorm1d(512),nn.Linear(in_features=512,out_features=1),nn.Sigmoid()).cuda()
    net = nn.DataParallel(nn.Sequential(res_net,classifier))
    net.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/linear_classifier_v1_500.pth'))
    net.eval()
    accuracy = 0
    print('Dataset Iteration')
    for n, (imager, imagef) in enumerate(dataloader):
        if n == 4250:
            break
        if (n+1)%100 == 0:
            print(n+1)
        with torch.no_grad():
            if net(imager.squeeze(0)) <= 0.5:
                accuracy+=1
            else:
                pass
            if net(imagef.squeeze(0))> 0.5:
                accuracy+=1
            else:
                pass
                
                
    print(accuracy/(4250*2))
    
        
    