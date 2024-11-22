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
import wandb
import random

import os
import time

wandb.init(
    # set the wandb project where this run will be logged
    project="Resnet CBAM Classifier",
    name = "run 2",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "CNN + CBAM",
    "dataset": "ELSA D3"
    },
    resume = 'allow',
    id = 'hnqcnja0'
)
print('wandb id: ' + str(wandb.run.id))

# loading the dataset and dataloader
dataset = DirectoryRandomDataset(dir='/work/cvcs2024/VisionWise/train')
dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=200,
    num_workers=8,
    dataset_mode=DirectoryRandomDataset.COUP
    )

checkpoint_name = 'checkpoint_rescbam_class_r1'

#cuda stuff
if not torch.cuda.is_available():
    raise RuntimeError('Cuda not available')
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

#learning stuff
model = nn.DataParallel(resnets.v2(freeze_resnet=True)).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0000001)
criterion = nn.BCEWithLogitsLoss()

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
    
    if (n + 1) % 5 == 0:
        optimizer.step()
        optimizer.zero_grad()

    if (n+1)%60==0:
        scheduler.step()
    if (n + 1) % 100 == 0: 
        try:
            print(f'Running loss at iter {n+1} is {(running_loss/100):.6f}')
            wandb.log({"loss": running_loss/100,"iter":(n+1)},step=(n+1))
        except:
            pass
        running_loss = 0.0
    
    if (n+1) % 100 == 0: 
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name,(n+1),optimizer,model)
        
        test_dataset = DirectorySequentialDataset('/work/cvcs2024/VisionWise/test')
        test_dataloader = TransformDataLoader(
        cropping_mode=RandomTransform.GLOBAL_CROP,
        dataset=test_dataset,
        batch_size=2,
        num_workers=8,
        dataset_mode=DirectoryRandomDataset.COUP,
        probability = 0.0
        )
        
        max_iter = 4266
        accuracy = 0.0
        with torch.no_grad():
            for n, (images, labels) in enumerate(test_dataloader):
                pred = model(images)
                pred = torch.round(pred)
                if pred[0]==labels[0]:
                    accuracy +=0.5
                if pred[1]==labels[1]:
                    accuracy +=0.5
                if n == max_iter:
                    break
        
                    
        wandb.log({"test_accuracy": accuracy}, step=(n + 1))
        print('Accuracy is -->' + str(accuracy*100/max_iter) + '%')
                
    



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