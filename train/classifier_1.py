import torch
import torch.nn as nn
from torch import optim
from models import classifiers , resnet_cbam
from data.datasets import DirectoryRandomDataset
from data.dataloader import TransformDataLoader
from utils.transform import RandomTransform
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
    classifier = nn.Sequential(nn.BatchNorm1d(512),nn.Linear(in_features=512,out_features=1),nn.Sigmoid())
    
    state_dict = torch.load('/work/cvcs2024/VisionWise/weights/res_weight_contrastive_v3_43000.pth',weights_only=True)
    state_dict = point_model_remover(state_dict=state_dict)
    
    embedder.load_state_dict(state_dict)
    for param in embedder.parameters():
        param.requires_grad = False
        
    model = nn.Sequential(embedder,classifier).cuda()
    model = nn.DataParallel(model)
    model.train()
    
    optimizer = optim.Adam(model.parameters(),lr=0.0001)    
    criterion = nn.BCELoss()
    
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP, dataset, batch_size=50,dataset_mode=DirectoryRandomDataset.COUP,num_workers=4,pacman=False)
    
    print('Let\'s train!!!!')
    for n , (images, labels) in enumerate(dataloader):

        
        out = model(images)
        labels = labels.to(torch.float32)

        
        loss = criterion(out,labels)
        loss.backward()
        
        print(f"Iteration {n + 1} --> Loss is {loss.item():.4f}", flush=True)
        if (n+1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        if (n+1) % 500 == 0:
            torch.save(model.state_dict(),f'/work/cvcs2024/VisionWise/weights/linear_classifier_v1_{n+1}.pth')
        
    
    
