import torch
import torch.nn as nn

import sys
from utils.modules import MultiCBAM
from model.iter_dataset import DirectoryRandomDataset
from utils.transform import RandomTransform
import torch.optim as optim
from model.dataloader import TransformDataLoader
import numpy as np

dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP,dataset,num_workers=8,probability=0.5)

cumulative_mean: list[float] =[]

print('starting!')
for index, (images, _) in enumerate(dataloader):
    cumulative_mean = ( ((index) * cumulative_mean) / (index + 1) ) + (torch.mean(images))
    if (index + 1)%100 == 0:
        print(f'mean of iteration: {index + 1} = {np.mean(cumulative_mean)}')