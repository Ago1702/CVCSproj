import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2
from data.datasets import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
from models import loss

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import resnet_cbam
from torch.amp import autocast, GradScaler
from utils import notifier

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #set this to false to debug
    torch.backends.cudnn.enabled=False

    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/test')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, dataset, batch_size=50,dataset_mode=DirectoryRandomDataset.COUP,num_workers=4,pacman=False)

    res_net = nn.DataParallel(resnet_cbam.v2().cuda())
    res_net.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v2_90000.pth', weights_only=True))
    
    running_loss = 0.0
    criterion = loss.ContrastiveLoss_V1()
    
    for n, (images, labels) in enumerate(dataloader):
        print(n)
        if n == 100:
            break

        out = res_net(images)
        loss = criterion(out,labels)
        running_loss +=loss.item()
        
    loss = running_loss/100