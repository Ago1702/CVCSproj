import torch
import torch.nn as nn
from torch import optim
from models import classifiers, resnet_cbam
from data.iter_dataset import DirectoryRandomDataset
from data.dataloader import TransformDataLoader
from utils.transform import RandomTransform
from torch.utils.data import TensorDataset
def point_model_remover(state_dict):
    '''
    The weights for the model were saved when it was wrapped by a nn.DataParallel.
    That means that the keys have an extra "module." part at the beginning.
    Use this function to remove it, allowing you to load the weights in a non-parallel network
    '''
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") 
        new_state_dict[new_key] = value
        
    return new_state_dict
        
        
if __name__ == '__main__':
    torch.backends.cudnn.enabled=False
    
    embedder = resnet_cbam.v2()
    # classifier = nn.Sequential(nn.BatchNorm1d(512),nn.Linear(in_features=512,out_features=1),nn.Sigmoid())
    classifier = classifiers.TransformerClassifier()

    state_dict = torch.load('/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v2_108000.pth',weights_only=True)
    state_dict = point_model_remover(state_dict=state_dict)
    
    embedder.load_state_dict(state_dict)
    
    for param in embedder.parameters():
        param.requires_grad = False
    
    model = nn.Sequential(embedder, classifier).cuda()
    model = nn.DataParallel(model)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, dataset, batch_size=50,
                                     dataset_mode=DirectoryRandomDataset.COUP, num_workers=4, pacman=False)
    
    print('Let\'s train!!!!')
    # metto in modalità eval l'embedder
    embedder.eval()

    embedding_list = []
    label_list = []

    # ottengo gli embedding per trainare il transformer
    with torch.no_grad():
        for images, labels in dataloader:
            embeddings = embedder(images)
            embedding_list.append(embeddings)
            label_list.append(labels)

    all_embeddings = torch.cat(embedding_list)
    all_labels = torch.cat(label_list)

    # concateno embedding e label
    embedding_dataset = TensorDataset(all_embeddings, all_labels)

    classifier_dataloader = TransformDataLoader(
        RandomTransform.GLOBAL_CROP,
        embedding_dataset,
        batch_size=50,
        dataset_mode=DirectoryRandomDataset.COUP,
        num_workers=4,
        pacman=False
    )
    # chaimo la funzione di train dlela classe TransformerClassifier
    classifier.train_model(train_loader=classifier_dataloader, num_epochs=2)


    # for n, (images, labels) in enumerate(dataloader):
    #
    #
    #     out = model(images)
    #     labels = labels.to(torch.float32)
    #
    #
    #     loss = criterion(out, labels)
    #     loss.backward()
    #
    #     print(f"Iteration {n + 1} --> Loss is {loss.item():.4f}", flush=True)
    #     if (n+1) % 10 == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
        
    
    
