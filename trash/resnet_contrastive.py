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

from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import resnet_cbam
from torch.amp import autocast, GradScaler
from utils import notifier
from utils.helpers import load_checkpoint , save_checkpoint


if __name__ == "__main__":
    experiment = 'checkpoint_v0_r1'
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da")
    
    #set this to false to debug
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark = True
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
    dataloader = TransformDataLoader(cropping_mode=RandomTransform.GLOBAL_CROP, dataset=dataset, batch_size=20,
                                     dataset_mode=DirectoryRandomDataset.COUP, num_workers=4, pacman=False)
    
    res_net = nn.DataParallel(resnet_cbam.v4().cuda())
    running_loss = 0.0

    criterion = loss.ContrastiveLoss_V1(margin=0.8)
    optimizer = optim.Adam(res_net.parameters(), lr = 0.00001)

    optimizer.zero_grad()
    
    start_index = load_checkpoint(experiment,optimizer,res_net)
    print(f'Loaded checkpoint number: {start_index}',flush=True)
    
    res_net.train()
    optimizer.zero_grad()

    print("Let's learn!", flush=True)
    
    for n, (images, labels) in enumerate(dataloader,start=start_index):

        out = res_net(images)
        iter_loss = criterion(out, labels.float())
        iter_loss.backward()
        
        running_loss +=iter_loss.item()

        print(f"Iteration {n + 1} --> Loss is {iter_loss.item():.6f}", flush=True)
        
        if (n + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
        

        if (n + 1) % 500 == 0: 
            try:
                notifier.send_notification(topic='current_disperazione',data=f"Running loss at iter {n+1} is {(running_loss/500):.6f}, validation loss is no-loss")
            except:
                pass
            running_loss = 0.0
        
        if (n+1) % 100 == 0:
            print(f'Saving checkpoint number: {n+1}',flush=True)
            save_checkpoint(experiment,(n+1),optimizer,res_net)
            
    
