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

from torchvision.transforms import ToPILImage

from models import resnets 
from models import loss
import wandb
import random
from signaling import wavelets

import os
import time

from info_nce import InfoNCE, info_nce

wandb.init(
    # set the wandb project where this run will be logged
    project="Resnet CBAM Contrastive With Wavelets",
    name = "run 1",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "CNN + CBAM (for real)",
    "dataset": "ELSA D3"
    },
    resume = 'allow'
)
print('wandb id: ' + str(wandb.run.id))

# loading the dataset and dataloader
dataset = DirectoryRandomDataset(dir='/work/cvcs2024/VisionWise/train')
dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=50,
    num_workers=2,
    dataset_mode=DirectoryRandomDataset.COUP,
    transform = wavelets.WaveletTransform()
    )

checkpoint_name = 'checkpoint_w_rescbam_contr_r1'

#cuda stuff
if not torch.cuda.is_available():
    raise RuntimeError('Cuda not available')
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

#learning stuff
model = nn.DataParallel(resnets.v7()).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = loss.ContrastiveLoss_V1(temperature=2)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.95)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
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
            wandb.log({"loss": running_loss/50,"iter":(n+1)},step=(n+1))
        except:
            pass
        running_loss = 0.0
    
    if (n+1) % 200 == 0: 
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name,(n+1),optimizer,model)
        scheduler.step()
    



'''
# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()'''
