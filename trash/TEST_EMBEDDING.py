import torch
from torch import nn as nn
from data.datasets import DirectorySequentialDataset
from data.datasets import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
import models.nets as nets
from utils.helpers import load_checkpoint
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

torch.cuda.manual_seed_all(42)
torch.backends.cudnn.enabled = False
checkpoint_name = 'ch_cbam50_contrastive'
model = nn.DataParallel(nets.cbam_classifier_50(drop_classifier=True)).cuda()
#model.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/ch_cbam50_classifier_2000.pth',weights_only=False)['model'])
print('loaded checkpoint:' + str(load_checkpoint(checkpoint_name,model=model)))

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
all_embeddings = []
all_labels = []
with torch.no_grad():
    accuracy = 0.0
    max_iter = 0
    print('Dataset Iteration')
    for test_images, test_labels in test_dataloader:
        out = F.normalize(model(test_images),p=2)

        all_embeddings.append(out.detach().cpu())  
        all_labels.append(test_labels.detach().cpu())
        
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
    plt.title(f"risultati iter cbam 50")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Classes')
    plt.savefig(os.path.expanduser(f'~/CVCSproj/outputs/{checkpoint_name}.png'))
    plt.close()
    all_embeddings.clear()
    all_labels.clear()

    
