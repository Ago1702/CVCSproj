import torch
import os
import os.path as path
import sys
import torch.optim as optim
from torch import nn as nn
from torchvision import models
from torchvision.transforms import v2
from data.datasets import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
from models import loss

from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import resnet_cbam
from torch.amp import autocast, GradScaler
from utils import notifier

def get_validation_loss(weights_file):
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da!")
    
    #set this to false to debug
    torch.backends.cudnn.enabled=False

    test_dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/test')
    test_dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, test_dataset, batch_size=50,dataset_mode=DirectoryRandomDataset.BASE,num_workers=8,pacman=False)

    test_net = nn.DataParallel(resnet_cbam.v2().cuda())
    test_net.load_state_dict(torch.load(weights_file, weights_only=True))
    
    test_running_loss = 0.0
    test_criterion = loss.ContrastiveLoss_V1(margin=0.5)
    
    for n, (images, labels) in enumerate(test_dataloader):
        print(n)
        if n == 160:
            break

        out = test_net(images)
        iter_loss_val = test_criterion(out,labels)
        test_running_loss +=iter_loss_val.item()
        
    test_running_loss = test_running_loss/160
    return test_running_loss

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"Memoria allocata GPU: {allocated:.2f} MB")
    print(f"Memoria riservata GPU: {reserved:.2f} MB")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Se non c'è cuda, lo prendi in cu..da")
    
    #set this to false to debug
    #torch.backends.cudnn.enabled=False
    
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
    dataloader = TransformDataLoader(cropping_mode=RandomTransform.GLOBAL_CROP, dataset=dataset, batch_size=50,
                                     dataset_mode=DirectoryRandomDataset.COUP, num_workers=4, pacman=False)
    
    res_net = nn.DataParallel(resnet_cbam.v2().cuda())
    running_loss = 0.0

    criterion = loss.ContrastiveLoss_V1(couple_boost=1.0)
    optimizer = optim.Adam(res_net.parameters(), lr = 0.001)
    scheduler = ExponentialLR(optimizer=optimizer,gamma=0.95)

    optimizer.zero_grad()

    start_index = 164500
    weights_file = f'/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v2_{start_index}.pth'

    if path.exists(weights_file):
        print("Loaded weights from file", flush=True)
        res_net.load_state_dict(torch.load(weights_file, weights_only=True))
        
    

    res_net.train()
    optimizer.zero_grad()
    # Initialize lists to store embeddings and labels
    all_embeddings = []
    all_labels = []
    print("Let's learn! (˶ᵔ ᵕ ᵔ˶)", flush=True)
    notifier.send_notification('step_disperazione',"Let's learn! (˶ᵔ ᵕ ᵔ˶)")
    #training parameters
    n_iter = 1
    
    for n, (images, labels) in enumerate(dataloader,start=start_index):

        for n_i in range(n_iter):

            out = res_net(images)
            iter_loss = criterion(out, labels.float())
            running_loss +=iter_loss.item()

            '''all_embeddings.append(out.detach().cpu())  # Move to CPU and detach from the graph
            all_labels.append(labels.detach().cpu())'''

            #clipping the gradients
            #torch.nn.utils.clip_grad_norm_(res_net.parameters(), max_norm=1.0)

            print(f"Iteration {n + 1} --> Loss is {iter_loss.item():.4f}", flush=True)
            if (n + 1) % 50 == 0:
                hard_batches = 0
                optimizer.step()
                optimizer.zero_grad()
                #print(f"Iteration {n + 1} --> Loss is {loss.item():.4f}", flush=True)
                #notifier.send_notification(topic='step_disperazione',data=f"Loss for {n+1} is {loss.item():.4f}")
            
            if (n+1) % 100 == 0:
                scheduler.step()

            if (n + 1) % 500 == 0: 
                notifier.send_notification(topic='current_disperazione',data=f"Running loss at iter {n+1} is {(running_loss/500):.4f}, validation loss is {get_validation_loss(weights_file=weights_file)}")
                running_loss = 0.0
            '''for name, param in res_net.named_parameters():
                if param.grad is not None:
                    print(f"Gradient of {name}: {param.grad}")
                else:
                    print(f"Gradient of {name}: {param.grad}")'''

            '''all_embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
            all_labels_np = torch.cat(all_labels, dim=0).flatten().numpy()

            # Step 2: Dimensionality Reduction
            tsne = TSNE(n_components=2, perplexity=10, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings_np)

            # Step 3: Plotting
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=all_labels_np, palette='viridis', s=100)
            plt.title(f"Iteration {n + 1}.{n_i + 1}")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.legend(title='Classes')
            plt.savefig(os.path.expanduser('~/CVCSproj/outputs/embedding.png'))
            plt.close()
            all_embeddings.clear()
            all_labels.clear()
            
            #print_gpu_memory_usage()'''
            torch.cuda.empty_cache()
            if (n+1) % 500 == 0:
                weights_file = f'/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v2_{n+1}.pth'
                torch.save(res_net.state_dict(), weights_file)
                
        
