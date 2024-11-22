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
from signaling import wavelets

from torch.amp import autocast, GradScaler
from utils import notifier
from torch.utils.data import DataLoader
from models.resnets import v7
import wandb
import time
from utils.helpers import state_dict_adapter

if __name__ == "__main__":
    #good iter ?? for run ??
    test_batch_size= 100
    iteration_index = 1000
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #uncomment this to false to debug
    torch.backends.cudnn.enabled=False

    dataset = DirectorySequentialDataset('/work/cvcs2024/VisionWise/test')
    dataloader = TransformDataLoader(
    cropping_mode=RandomTransform.GLOBAL_CROP,
    dataset=dataset,
    batch_size=test_batch_size,
    num_workers=2,
    dataset_mode=DirectoryRandomDataset.COUP,
    transform = wavelets.WaveletTransform(),
    center_crop=True
    )


    embedder = v7()
    embedder.load_state_dict(state_dict_adapter(torch.load(f'/work/cvcs2024/VisionWise/weights/checkpoint_w_rescbam_contr_r1_{iteration_index}.pth',weights_only=False)['model'],string_to_remove='module.'))
    embedder=nn.DataParallel(embedder.cuda())
    

    max_iter = 4250
    accuracy = 0.0
    n=0

    all_embeddings = []
    all_labels = []
    
    print('Dataset Iteration')
    for images, labels in dataloader:
        print(str(n))
        with torch.no_grad():
            pred = embedder(images)
            all_embeddings.append(pred.detach().cpu()) 
            all_labels.append(labels.detach().cpu())
        n+=test_batch_size
                
    print('end of iteration',flush=True)
    all_embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).flatten().numpy()
    print('step 2',flush=True)
    # Step 2: Dimensionality Reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings_np)

    print('step 3',flush=True)
    # Step 3: Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=all_labels_np, palette='viridis', s=100)
    plt.title("risultati" + f" run 1 iter {iteration_index}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Classes')
    plt.savefig(os.path.expanduser(f'~/CVCSproj/outputs/rescbam_contr_wavelet_r1_{iteration_index}.png'))
    plt.close()
    all_embeddings.clear()
    all_labels.clear()

    