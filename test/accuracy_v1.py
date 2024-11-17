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

import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from torch.amp import autocast, GradScaler
from utils import notifier
from torch.utils.data import DataLoader
from models.resnets import v1
import wandb
import time

if __name__ == "__main__":
    iteration_index = 800
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #uncomment this to false to debug
    torch.backends.cudnn.enabled=False

    dataset = DirectorySequentialDataset('/work/cvcs2024/VisionWise/train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)


    classifier = v1()
    classifier.load_state_dict(torch.load(f'/work/cvcs2024/VisionWise/weights/checkpoint_res_class_r4_{iteration_index}.pth')['model'])
    #classifier=nn.Sequential(classifier,nn.Sigmoid()).cuda()
    


    accuracy = 0
    print('Dataset Iteration')
    for n, (imager, imagef) in enumerate(dataloader):
        if n == 300:
            break
        if (n+1)%100 == 0:
            print(n+1)
        with torch.no_grad():
            predr = classifier(imager.squeeze(0))
            predf = classifier(imagef.squeeze(0))
            print('real --> ' + str(predr))
            print('fake --> ' + str(predf))

            if  predr<= 0.5:
                accuracy+=1
            else:
                pass
            if predf> 0.5:
                accuracy+=1
            else:
                pass
                
                
    print(accuracy)
    
        
    