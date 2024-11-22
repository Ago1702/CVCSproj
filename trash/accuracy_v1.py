import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2 as transforms
from data.datasets import DirectoryRandomDataset
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
from utils.helpers import state_dict_adapter

if __name__ == "__main__":
    #good iter = 10000
    test_batch_size= 100
    iteration_index = 10000
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #uncomment this to false to debug
    torch.backends.cudnn.enabled=False

    dataset = DirectorySequentialDataset('/work/cvcs2024/VisionWise/test')
    dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=test_batch_size,
    num_workers=8,
    dataset_mode=DirectoryRandomDataset.COUP,
    probability = 0.0,
    center_crop=True
    )


    classifier = v1()
    classifier.load_state_dict(torch.load(f'/work/cvcs2024/VisionWise/weights/checkpoint_res_class_r4_{iteration_index}.pth',weights_only=False)['model'])
    classifier=nn.Sequential(classifier,nn.Sigmoid()).cuda()
    

    max_iter = 4250
    accuracy = 0.0
    n=0
    
    print('Dataset Iteration')
    for images, labels in dataloader:
        print(str(n))
        with torch.no_grad():
            pred = classifier(images)
            pred = torch.round(pred)
            good_answers = torch.sum(pred == labels)/2
            accuracy+=good_answers.item()
        n+=test_batch_size
                
                
    print('Accuracy is -->' + str(accuracy*100/max_iter) + '%')




#ACC AT ITER 10'000 = 95.5764705882353%
    
        
    