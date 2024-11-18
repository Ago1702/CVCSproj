import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np
from info_nce import InfoNCE, info_nce

def is_partner(i:int,j:int,labels: torch.tensor)->bool:
    '''
    Ultra specific helper funcion
    It helps to determine if two indexes from a batch from model + dataloader + dataset are from a couple.

    It only works with RandomDirectoryDataset + TransformDataloader + a model that keeps the batch order

    EX: 
        labels:
        0 -->couple 0
        1 -->couple 0
        0 -->couple 1
        1 -->couple 1 
        0 -->couple 2
        1 -->couple 2
    '''

    if labels[i]==0:
        return i==(j-1)
    elif labels[i]==1:
        return i==(j+1)
    else:
        raise RuntimeError(f'Invalid label found:{labels[i]}')
class ContrastiveLoss_V1(nn.Module):
    def __init__(self,couple_boost:float = 1.0,margin:float = 1.0,temperature:float = 1.0):
        super(ContrastiveLoss_V1,self).__init__()
        self.couple_boost=couple_boost
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings:torch.Tensor,labels:torch.Tensor):
        '''
        Args: 
            embeddings (torch.Tensor): a tensor with two dimensions: (n_samples,embedding_size)
            labels (torch.Tensor): a tensor with two dimensions: (n_samples,1)
        '''
        #label == 0 --> real
        #label == 1 --> fake
        loss = 0.0
        embeddings = F.normalize(embeddings,p=2,dim=1,eps=1e-6)
        
        n_of_comparisons = 0
        for i in range(embeddings.size(0)):
            sample = embeddings[i,:]
            for j in range(embeddings.size(0)):
                if i <= j:
                    continue
                distance = F.pairwise_distance(embeddings[i], embeddings[j])
                if labels[i] == labels[j]:
                    positive_loss = 0.5 * torch.pow(torch.clamp(distance - 0.1, min= 0), 2)
                    if torch.isnan(positive_loss):
                        print(f"NaN detected in positive loss for samples {i} and {j}")
                        print(embeddings[i])
                        print(embeddings[j])
                    loss +=positive_loss
                    n_of_comparisons += 1
                else:
                    p_couple_boost: float = 1.0
                    if is_partner(i,j,labels=labels):
                        p_couple_boost= self.couple_boost #this is not == 1 only when they are "partners" (simple strategy to keep code simplier)
                    negative_loss = 0.5 * p_couple_boost*torch.pow(torch.clamp(self.margin - distance, min= 0), 2)
                    if torch.isnan(negative_loss):
                        print(f"NaN detected in negative loss for samples {i} and {j}")
                    loss += negative_loss
                    n_of_comparisons += 1
        return loss/n_of_comparisons
        

class ContrastiveLoss_V2(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss_V2, self).__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        '''
        Args: 
            embeddings (torch.Tensor): a tensor with two dimensions: (n_samples, embedding_size)
            labels (torch.Tensor): a tensor with one dimension: (n_samples)
        '''
        # label == 0 --> real, label == 1 --> fake
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-6)

        # Compute pairwise distances in a vectorized way
        distances = torch.cdist(embeddings, embeddings, p=2) + 1e-6  # Shape: (n_samples, n_samples)

        # Create masks for positive (same class) and negative (different class) pairs
        labels = labels.view(-1, 1)  # Make labels shape (n_samples, 1) for broadcasting
        positive_mask = (labels == labels.T).float() - torch.eye(labels.size(0), device=labels.device)
        negative_mask = (labels != labels.T).float()

        # Calculate positive loss for same-class pairs
        positive_loss = torch.pow(distances, 2) * positive_mask
        positive_loss = positive_loss.sum() / positive_mask.sum().clamp(min=1)

        # Calculate negative loss for different-class pairs
        negative_loss = torch.pow(torch.clamp(self.margin - distances, min=1e-6), 2) * negative_mask
        negative_loss = negative_loss.sum() / negative_mask.sum().clamp(min=1)

        # Total loss
        loss = positive_loss + 0.5 * negative_loss
        return loss
    
class ContrastiveLoss_V3(nn.Module):
    def __init__(self,margin:float = 1.0):
        super(ContrastiveLoss_V3,self).__init__()
        self.margin = margin

    def forward(self, embeddings:torch.Tensor,labels:torch.Tensor):
        '''
        Args: 
            embeddings (torch.Tensor): a tensor with two dimensions: (n_samples,embedding_size)
            labels (torch.Tensor): a tensor with two dimensions: (n_samples,1)
        '''
        #label == 0 --> real
        #label == 1 --> fake
        loss = 0.0
        embeddings = F.normalize(embeddings,p=2,dim=1,eps=1e-6)
        
        n_of_comparisons = 0
        for anchor in range(embeddings.size(0)):
            positive = 0
            negative = 0
            while True:
                positive=np.random.choice(range(embeddings.size(0)))
                if anchor!=positive and labels[anchor]==labels[positive]:
                    break
            while True:
                negative=np.random.choice(range(embeddings.size(0)))
                if labels[anchor]!=labels[negative]:
                    break
            distance_pos = F.pairwise_distance(embeddings[anchor,:], embeddings[positive,:]) + 1e-6
            distance_neg = F.pairwise_distance(embeddings[anchor,:], embeddings[negative,:]) + 1e-6
            
            positive_loss = torch.pow(distance_pos, 2)
            negative_loss = torch.pow(torch.clamp(self.margin - distance_neg, min= 1e-6), 2)
            
            loss+=positive_loss
            loss+=negative_loss
            
            n_of_comparisons += 2
        
        return loss/n_of_comparisons
    

class ContrastiveLoss_V4(nn.Module):
    def __init__(self,margin:float = 1.0):
        super(ContrastiveLoss_V4,self).__init__()
        self.margin = margin

    def forward(self, embeddings:torch.Tensor,labels:torch.Tensor):
        '''
        Args: 
            embeddings (torch.Tensor): a tensor with two dimensions: (n_samples,embedding_size)
            labels (torch.Tensor): a tensor with two dimensions: (n_samples,1)
        '''
        #label == 0 --> real
        #label == 1 --> fake
        loss = 0.0
        embeddings = F.normalize(embeddings,p=2,dim=1,eps=1e-6)
        
        n_of_comparisons = 0
        for anchor in range(embeddings.size(0)):
            #loss with one randomly chosen positive sample
            while True:
                positive=np.random.choice(range(embeddings.size(0)))

                if anchor!=positive and labels[anchor]==labels[positive]:
                    distance_pos = F.pairwise_distance(embeddings[anchor,:], embeddings[positive,:])
                    positive_loss = torch.pow(distance_pos, 2)
                    loss+=positive_loss
                    n_of_comparisons+=1
                    break
            
            #loss for all negative samples
            negative_comparisons = 0
            indices = torch.randperm(embeddings.size(0))
            for sample in indices:
                if labels[anchor]!=labels[sample]:
                    distance_neg = F.pairwise_distance(embeddings[anchor,:], embeddings[sample,:])
                    negative_loss = torch.pow(torch.clamp(self.margin - distance_neg, min= 0), 2)
                    loss+=negative_loss
                    negative_comparisons+=1
            n_of_comparisons+=negative_comparisons
        
        return loss/n_of_comparisons
    
class InfoNCE_complete_V1(nn.Module):
    def __init__(self):
        super(InfoNCE_complete_V1,self).__init__()

    def forward(self, embeddings:torch.Tensor,labels:torch.Tensor):
        '''
        Args: 
            embeddings (torch.Tensor): a tensor with two dimensions: (n_samples,embedding_size)
            labels (torch.Tensor): a tensor with two dimensions: (n_samples,1)
        '''
        #label == 0 --> real
        #label == 1 --> fake
        out_loss = 0.0
        embeddings = F.normalize(embeddings,p=2,dim=1)
        N = embeddings.size(0)
        
        for anchor_idx in range(N):
            anchor_loss = InfoNCE(negative_mode='unpaired')
            
            pos_indices = torch.where((labels == labels[anchor_idx]) & (torch.arange(N).cuda() != anchor_idx))[0]
            if len(pos_indices) > 0:
                #random selection of a positive sample
                pos_idx = pos_indices[torch.randint(len(pos_indices), (1,)).item()]
                positive = embeddings[pos_idx]
            else:
                #impossible case
                raise ValueError(f"No positive sample found for anchor index {anchor_idx}.")
            
            #getting all the negative samples
            neg_indices = torch.where(labels != labels[anchor_idx])[0]
            negatives = embeddings[neg_indices]
            out_loss += anchor_loss(embeddings[anchor_idx].unsqueeze(0),positive.unsqueeze(0),negatives)
            
        
        return out_loss/N
