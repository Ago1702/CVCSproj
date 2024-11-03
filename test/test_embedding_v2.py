import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2
from data.iter_dataset import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
from models import loss

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import resnet_cbam
from torch.amp import autocast, GradScaler
from utils import notifier

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #set this to false to debug
    torch.backends.cudnn.enabled=False

    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/test')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, dataset, batch_size=50,dataset_mode=DirectoryRandomDataset.COUP,num_workers=4,pacman=False)

    res_net = nn.DataParallel(resnet_cbam.v2().cuda())
    res_net.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v2_70000.pth', weights_only=True))

    all_embeddings = []
    all_labels = []

    for n, (images, labels) in enumerate(dataloader):
        print(n)
        if n == 100:
            break

        out = res_net(images)
        all_embeddings.append(out.detach().cpu())  # Move to CPU and detach from the graph
        all_labels.append(labels.detach().cpu())

    all_embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).flatten().numpy()
    print('step 2')
    # Step 2: Dimensionality Reduction
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings_np)

    print('step 3')
    # Step 3: Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=all_labels_np, palette='viridis', s=100)
    plt.title(f"risultati")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Classes')
    plt.savefig(os.path.expanduser('~/CVCSproj/outputs/embedding_2.png'))
    plt.close()
    all_embeddings.clear()
    all_labels.clear()