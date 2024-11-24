from data.dataloader import TransformDataLoader
from data.datasets import DirectoryRandomDataset
from data.datasets import DirectorySequentialDataset

from utils.transform import RandomTransform
from utils.helpers import save_checkpoint
from utils.helpers import load_checkpoint

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.transforms import ToPILImage

from models.loss import ContrastiveLoss_V1
from models import nets
import wandb


wandb.init(
    # set the wandb project where this run will be logged
    project="CBAM-152-Contrastive",
    name = "run 1",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.00001,
    "architecture": "CNN + CBAM",
    "dataset": "ELSA D3"
    },
    resume = 'allow'
)
print('wandb run id: ' + str(wandb.run.id))

# loading the dataset and dataloader
dataset = DirectoryRandomDataset(dir='/work/cvcs2024/VisionWise/train')
dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=100,
    num_workers=8,
    dataset_mode=DirectoryRandomDataset.COUP
    )

checkpoint_name = 'ch_cbam152_contrastive'

#cuda stuff
if not torch.cuda.is_available():
    raise RuntimeError('Cuda not available')
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

#learning stuff
model = nn.DataParallel(nets.cbam_classifier_152(freeze_mode='none',drop_classifier=True)).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=0.90)
criterion = ContrastiveLoss_V1(margin=1.0,temperature=2)

if load_checkpoint(checkpoint_name=checkpoint_name) == 0:
    print('Loaded starting checkpoint: ' + str(load_checkpoint(checkpoint_name = 'ch_cbam152_classifier',model=model)),flush=True)
    start_index = 0
else:
    #loading previous state
    start_index = load_checkpoint(checkpoint_name,optimizer,model)
    print(f'Loaded checkpoint number: {start_index}',flush=True)

running_loss = 0.0

for n, (images, labels) in enumerate(dataloader,start=start_index):
    model.train()
    out = model(images)
    iter_loss = criterion(out, labels.float())
    iter_loss.backward()
    
    running_loss +=iter_loss.item()

    print(f"Iteration {n + 1} --> Loss is {iter_loss.item():.6f}", flush=True)
    
    #stepping to learn
    if (n + 1) % 5 == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    if (n + 1) % 50 == 0:
        scheduler.step()
        
    #logging data on wandb
    if (n + 1) % 50 == 0: 
        try:
            print(f'Running loss at iter {n+1} is {(running_loss/50):.6f}')
            wandb.log({"training_loss": running_loss/50},step=(n+1))
        except:
            pass
        running_loss = 0.0
    
    if (n+1) % 500 == 0: 
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name,(n+1),optimizer,model)
    
                
    if(n+1)%5000 == 0:
        wandb.finish()
        break


