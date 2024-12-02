from data.dataloader import TransformDataLoader
from data.datasets import DirectoryRandomDataset
from data.datasets import DirectorySequentialDataset

from utils.transform import RandomTransform
from utils.helpers import save_checkpoint
from utils.helpers import load_checkpoint

from signaling.wavelets import WaveletTransform

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.transforms import ToPILImage

#from models import nets
from signaling.signalnet import SignalNet
import wandb


SCHED = 100
OPT = 5
EVAL = 300
SAVE = 900
SEND = 50
WAND = True
UNFREEZE = 600
SCHED_START = 2100


if WAND:
    wandb.init(
        project="Wavelet-Classifier",
        name = "run 2",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.0005,
        "architecture": "Signalnet",
        "dataset": "ELSA D3"
        },
        resume = 'allow'
    )

    print('wandb run id: ' + str(wandb.run.id))

dataset = DirectoryRandomDataset(dir='/work/cvcs2024/VisionWise/train')
dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=100,
    num_workers=4,
    dataset_mode=DirectoryRandomDataset.BASE
    )

checkpoint_name = 'wave50_classifier'
checkpoint_name_sace = 'wave50trans_classifier'

if not torch.cuda.is_available():
    raise RuntimeError('Cuda not available')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

model = SignalNet(39)
model.unfreeze()
model = nn.DataParallel(model).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

ciao = load_checkpoint(checkpoint_name=checkpoint_name,model=model)
print(ciao)
print(f'Loaded checkpoint number: {0}',flush=True)


optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=0.95)
criterion = nn.BCEWithLogitsLoss()

running_loss = 0.0

for n, (images, labels) in enumerate(dataloader,start=0):
    model.train()
    #torch.use_deterministic_algorithms(False)
    out = model(images)
    iter_loss = criterion(out, labels.float())
    iter_loss.backward()
    
    running_loss +=iter_loss.item()

    print(f"Iteration {n + 1} --> Loss is {iter_loss.item():.6f}", flush=True)
    
    #stepping to learn
    if (n + 1) % OPT == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    if (n + 1) % SCHED == 0 and (n + 1) > SCHED_START:
        scheduler.step()
        
    #logging data on wandb
    if (n + 1) % SEND == 0: 
        try:
            print(f'Running loss at iter {n+1} is {(running_loss/SEND):.6f}', flush=True)
            if(WAND):
                wandb.log({"training_loss": running_loss/SEND},step=(n+1))
        except:
            pass
        running_loss = 0.0
    
    if (n+1) % SAVE == 0: 
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name_sace,(n+1),optimizer,model)
    
    if (n+1) % EVAL == 0:
        #torch.use_deterministic_algorithms(True)
        #dataset and dataloader for testing
        test_dataset = DirectorySequentialDataset(dir='/work/cvcs2024/VisionWise/test', beaviour=DirectoryRandomDataset.BASE)
        test_dataloader = TransformDataLoader(
            cropping_mode=RandomTransform.GLOBAL_CROP,
            dataset=test_dataset,
            batch_size=100,
            num_workers=4,
            dataset_mode=DirectoryRandomDataset.BASE,
            probability=0.0
            )
        
        model.eval()
        with torch.no_grad():
            accuracy = 0.0
            max_iter = 0
            print('Dataset Iteration', flush=True)
            for test_images, test_labels in test_dataloader:
                with torch.no_grad():
                    test_pred = torch.sigmoid(model(test_images))
                    test_pred = torch.round(test_pred)
                    max_iter += test_pred.shape[0]
                    good_answers = torch.sum(test_pred == test_labels)
                    accuracy+=good_answers.item()  
            try:
                if(WAND):    
                    wandb.log({"test_accuracy": accuracy*100/max_iter}, step=(n + 1))
            except:
                pass
            print('Accuracy is -->' + str(accuracy*100/max_iter) + '%', flush=True)

    if (n + 1) == UNFREEZE:
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name,(n+1),optimizer,model, path="/homes/dagostini/CVCSproj/ueits")
        for param in model.parameters():
            param.requires_grad_(True)