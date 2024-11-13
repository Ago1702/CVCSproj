import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2
from data.datasets import DirectorySequentialDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
from models import loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import resnet_cbam
from torch.amp import autocast, GradScaler
from utils import notifier

if __name__ == "__main__":
    iteration_index = 300
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #uncomment this to false to debug
    #torch.backends.cudnn.enabled=False

    dataset = DirectorySequentialDataset('/work/cvcs2024/VisionWise/test')
    transform = RandomTransform(p=1)
    
    res_net = nn.DataParallel(resnet_cbam.v2().cuda())
    res_net.load_state_dict(torch.load(f'/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v4_{iteration_index}.pth', weights_only=True))
    res_net.eval()
    all_embeddings = []
    all_labels = []
    
    print('Dataset Iteration')
    for n, (imager, imagef) in enumerate(dataset):
        if n == 4000:
            break
        
        if (n+1) % 50 == 0:
            print(n+1,flush=True)
        
        out_r = F.normalize(res_net(transform(imager.unsqueeze(0).cuda())),p=2)
        out_f = F.normalize(res_net(transform(imagef.unsqueeze(0).cuda())),p=2)
        all_embeddings.append(out_r.detach().cpu())  # Move to CPU and detach from the graph
        all_embeddings.append(out_f.detach().cpu())  # Move to CPU and detach from the graph
        all_labels.append(torch.tensor([0],dtype=torch.long))
        all_labels.append(torch.tensor([1],dtype=torch.long))
        
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
    plt.title(f"risultati iter {iteration_index}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Classes')
    plt.savefig(os.path.expanduser(f'~/CVCSproj/outputs/embedding_{iteration_index}.png'))
    plt.close()
    all_embeddings.clear()
    all_labels.clear()