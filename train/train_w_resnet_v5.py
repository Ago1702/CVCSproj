from data.dataloader import TransformDataLoader
from data.datasets import DirectoryRandomDataset
from data.datasets import DirectorySequentialDataset

from utils.transform import RandomTransform
from utils.helpers import save_checkpoint
from utils.helpers import load_checkpoint
from utils.helpers import save_tensor_to_png

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.transforms import ToPILImage

from models import resnets 
from signaling import wavelets
import wandb
import random

import os
import time

from info_nce import InfoNCE, info_nce
wandb.init(
    # set the wandb project where this run will be logged
    project="Resnet Wavelet Classifier",
    name = "run 1",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "CNN",
    "dataset": "ELSA D3"
    }
)

# loading the dataset and dataloader
dataset = DirectoryRandomDataset(dir='/work/cvcs2024/VisionWise/train')
dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=20,
    num_workers=2,
    dataset_mode=DirectoryRandomDataset.COUP,
    transform = wavelets.WaveletTransform()
    )

checkpoint_name = 'checkpoint_w_res_class_r1'
torch.backends.cudnn.enabled = False

#cuda stuff
if not torch.cuda.is_available():
    raise RuntimeError('Cuda not available')
torch.backends.cudnn.benchmark = True

#learning stuff
model = nn.DataParallel(resnets.v5()).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
scheduler = ExponentialLR(optimizer=optimizer,gamma = 0.95)

#loading previous state
start_index = load_checkpoint(checkpoint_name,optimizer,model)
print(f'Loaded checkpoint number: {start_index}',flush=True)
model.train()

running_loss = 0.0

for n, (images, labels) in enumerate(dataloader,start=start_index):
    out = model(images)
    iter_loss = criterion(out, labels.float())
    iter_loss.backward()
    
    running_loss +=iter_loss.item()

    print(f"Iteration {n + 1} --> Loss is {iter_loss.item():.6f}", flush=True)
    
    if (n + 1) % 1 == 0:
        optimizer.step()
        optimizer.zero_grad()
    

    if (n + 1) % 50 == 0: 
        try:
            print(f'Running loss at iter {n+1} is {(running_loss/50):.6f}')
            lr= None
            for param_group in optimizer.param_groups:
                lr=param_group['lr']
            wandb.log({"loss": running_loss/50,"iter":(n+1),"lr":lr})
        except:
            pass
        running_loss = 0.0
    
    if (n+1) % 100 == 0: 
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name,(n+1),optimizer,model)
        scheduler.step()
            
    