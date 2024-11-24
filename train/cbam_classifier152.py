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

from models import nets
import wandb

#2000 good iter

wandb.init(
    # set the wandb project where this run will be logged
    project="CBAM-152-Classifier",
    name = "run 1",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
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

checkpoint_name = 'ch_cbam152_classifier'

#cuda stuff
if not torch.cuda.is_available():
    raise RuntimeError('Cuda not available')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

#learning stuff
model = nn.DataParallel(nets.cbam_classifier_152(freeze_backbone=False,drop_backbone = True)).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
scheduler = ExponentialLR(optimizer=optimizer,gamma=0.95)
criterion = nn.BCEWithLogitsLoss()  

#loading previous state
start_index = load_checkpoint(checkpoint_name,optimizer,model)
print(f'Loaded checkpoint number: {start_index}',flush=True)

running_loss = 0.0

for n, (images, labels) in enumerate(dataloader,start=start_index):
    model.train()
    #torch.use_deterministic_algorithms(False)
    out = model(images)
    iter_loss = criterion(out, labels.float())
    iter_loss.backward()
    
    running_loss +=iter_loss.item()

    print(f"Iteration {n + 1} --> Loss is {iter_loss.item():.6f}", flush=True)
    
    #stepping to learn
    if (n + 1) % 1 == 0:
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
    
    if (n+1) % 1000 == 0: 
        print(f'Saving checkpoint number: {n+1}',flush=True)
        save_checkpoint(checkpoint_name,(n+1),optimizer,model)
    
    if (n+1) % 200 == 0:
        #torch.use_deterministic_algorithms(True)
        #dataset and dataloader for testing
        test_dataset = DirectorySequentialDataset(dir='/work/cvcs2024/VisionWise/test')
        test_dataloader = TransformDataLoader(
            cropping_mode=RandomTransform.GLOBAL_CROP,
            dataset=test_dataset,
            batch_size=100,
            num_workers=4,
            dataset_mode=DirectoryRandomDataset.COUP,
            probability=0.0,
            center_crop=True
            )
        model.eval()
        with torch.no_grad():
            accuracy = 0.0
            max_iter = 0
            print('Dataset Iteration')
            for test_images, test_labels in test_dataloader:
                with torch.no_grad():
                    test_pred = torch.sigmoid(model(test_images))
                    test_pred = torch.round(test_pred)
                    max_iter += test_pred.shape[0]
                    good_answers = torch.sum(test_pred == test_labels)
                    accuracy+=good_answers.item()
                
            try:    
                wandb.log({"test_accuracy": accuracy*100/max_iter}, step=(n + 1))
            except:
                pass
            print('Accuracy is -->' + str(accuracy*100/max_iter) + '%')
                
    if(n+1)%10000 == 0:
        wandb.finish()
        break


