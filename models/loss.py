import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
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
    def __init__(self,couple_boost:float = 1.0,margin:float = 1.0):
        super(ContrastiveLoss_V1,self).__init__()
        self.couple_boost=couple_boost
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
        for i in range(embeddings.size(0)):
            sample = embeddings[i,:]
            for j in range(embeddings.size(0)):
                if i <= j:
                    continue
                distance = F.pairwise_distance(embeddings[i], embeddings[j]) + 1e-6
                if labels[i] == labels[j]:
                    positive_loss = torch.pow(distance, 2)
                    if torch.isnan(positive_loss):
                        print(f"NaN detected in positive loss for samples {i} and {j}")
                        print(embeddings[i])
                        print(embeddings[j])
                    loss +=positive_loss
                    n_of_comparisons += 1
                else:
                    p_couple_boost: float = 1.0
                    if is_partner(i,j,labels=labels):
                        p_couple_boost= self.couple_boost #this is not 1 only when they are "partners" (simple strategy to keep code simplier)
                    negative_loss = 0.5 * p_couple_boost*torch.pow(torch.clamp(self.margin - distance, min= 1e-6), 2)
                    if torch.isnan(negative_loss):
                        print(f"NaN detected in negative loss for samples {i} and {j}")
                    loss += negative_loss
                    n_of_comparisons += 1
        return loss/n_of_comparisons
            
class ContrastiveLoss_V2(nn.Module):
    def __init__(self,couple_boost:float = 1.0,margin:float = 1.0):
        super(ContrastiveLoss_V2,self).__init__()
        self.couple_boost=couple_boost
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
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-6)
        
        # Compute pairwise distances matrix
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Mask for positive (same label) and negative pairs (different labels)
        pos_mask = labels == labels
        neg_mask = ~pos_mask

        # Apply the couple boost only to specific pairs (partners)
        partner_mask = torch.zeros_like(distances, dtype=torch.bool)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and is_partner(i, j, labels):  # Check if they are partners
                    partner_mask[i, j] = True

        # Loss for positive pairs
        positive_loss = torch.pow(distances, 2) * pos_mask.float()
        
        # Loss for negative pairs, with clamping and couple boost
        negative_loss = 0.5 * torch.pow(torch.clamp(self.margin - distances, min=1e-6), 2)
        negative_loss[partner_mask] *= self.couple_boost  # Apply couple boost only to partner pairs
        negative_loss *= neg_mask.float()

        # Total loss computation, divided by the number of comparisons
        loss = positive_loss.sum() + negative_loss.sum()
        n_of_comparisons = pos_mask.sum() + neg_mask.sum()
        
        return loss / n_of_comparisons
        