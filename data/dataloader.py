import torch
import os
from PIL import Image
from torch.utils.data import DataLoader
from data.datasets import DirectoryRandomDataset
from data.datasets import DirectorySequentialDataset
from utils.transform import RandomTransform
from torchvision.utils import save_image
import gc

class TransformDataLoader(DataLoader):
    '''
    A dataloader that transforms the images

    This class takes a batch of images provided by a DirectoryRandomDataset, transforms them using a 
    RandomTransform (either LOCAL or GLOBAL), then puts them in a 4 dimensional tensor of dimensions:
    (batch_size, num_channels, height, width)
    The dataloader automatically moves the files onto the cuda device, checking its presence during initialization.

    Usage: call the constructor, then iterate over the object. Each time you iterate you get two tensors:
    The first tensor is a batch of batch_size * images
    The second tensor is the tensor of the labels

    Methods:
        custom_collate(batch) : a function declared as a method to be class aware. stacks images into a unique tensor

    '''
    def __init__(self, cropping_mode:int ,dataset:DirectoryRandomDataset, num_workers: int,dataset_mode:int, batch_size:int =32,num_channels:int = 3
                 , probability: float = 0.5, pacman : bool = False
                 ):
        
        '''
        Initializes the transform data loader

        Args:
            cropping_mode (int) : it will be passed to the RandomTransform. Must be either RandomTransform.LOCAL_CROP or RandomTransform.GLOBAL_CROP
            dataset (DirectoryRandomDataset) : the dataset class. Support for different dataset classes may be added in the future.
            num_workers (int) : the number of workers that the superclass will automatically
            dataset_mode (int) : one of RandomDirectoryDataset.COUP or RandomDirectoryDataset.BASE
            batch_size (int) : first dimension of the images tensor that the dataloader will return. MUST be an even number
            num_channels (int) : second dimension of the images tensor that the dataloader will return
            probability (float) : probability that a transformation will be applied. It will be passed to the RandomTransform constructor
        Attributes: 
            tuple[torch.Tensor , torch.Tensor] : a tuple, containing the stacked images and the stacked labels
        '''
        #error handling
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available (????????)')
        if cropping_mode != RandomTransform.GLOBAL_CROP and cropping_mode != RandomTransform.LOCAL_CROP:
            raise RuntimeError(f'TransformDataLoader was called with an invalid cropping_mode: {cropping_mode} .Look at the TransformDataLoader documentation.')
        if dataset_mode not in [DirectoryRandomDataset.BASE,DirectoryRandomDataset.COUP]:
            raise RuntimeError(f'Invalid dataset_mode: {dataset_mode}')
        if batch_size%2 != 0:
            raise RuntimeError(f'batch_size must be even')

        dataset.change_mode(dataset_mode)

        if dataset_mode ==DirectoryRandomDataset.COUP:
            batch_size=int(batch_size/2)

        super().__init__(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.custom_collate)
        self.transform = RandomTransform(cropping_mode=cropping_mode,p=probability,pacman=pacman)
        self.cropping_mode = cropping_mode
        self.num_channels = num_channels
        self.dataset_mode = dataset_mode


    def custom_collate(self,batch):
        '''
        A custom version of the collate function.

        This method is a custom version of the default one. Its role is taking a list of samples from the dataset, transforming and stacking them.

        Args:
            batch (list[tuple[torch.Tensor , torch.Tensor]]) : the batch, in which the first tensor is the image, and the second is the label (in base mode)

        Returns:
            tuple[torch.Tensor , torch.Tensor] : a tuple, containing the stacked images and the stacked labels
        '''

        
        crop_size: tuple[int,int] = (0,0)
        if self.cropping_mode == RandomTransform.LOCAL_CROP:
            crop_size = RandomTransform.LOCAL_CROP_SIZE
        elif self.cropping_mode == RandomTransform.GLOBAL_CROP:
            crop_size = RandomTransform.GLOBAL_CROP_SIZE
        else:
            raise RuntimeError("TransformDataLoader found an invalid cropping_mode = ({cropping_mode[0]},{cropping_mode[1]}) during collate function execution.")

        x_batch_list: list[torch.Tensor] = []
        y_batch_list: list[torch.Tensor] = []
        
    
        if self.dataset_mode == DirectoryRandomDataset.BASE:
            for image, label in batch:
                x_batch_list.append(self.transform(image))
                y_batch_list.append(label)
                gc.collect()

        elif self.dataset_mode == DirectoryRandomDataset.COUP:
            for couple in batch:
                x_batch_list.append(self.transform(couple[0])) #real
                x_batch_list.append(self.transform(couple[1])) #fake
                y_batch_list.append(torch.tensor(0, dtype=torch.long))
                y_batch_list.append(torch.tensor(1, dtype=torch.long))
                gc.collect()

        x_batch_tensor: torch.Tensor = torch.stack(x_batch_list).squeeze(1)
        y_batch_tensor: torch.Tensor = torch.stack(y_batch_list)

        x_min = torch.min(x_batch_tensor)
        x_max = torch.max(x_batch_tensor)
        gc.collect()
        
        x_batch_tensor = (x_batch_tensor - x_min) / (x_max - x_min)

        return x_batch_tensor , y_batch_tensor
    
    def __iter__(self):
        for batch in super().__iter__():
            yield torch.clamp(batch[0].cuda(),min = 0.0,max = 1.0) , batch[1].unsqueeze(1).cuda() 
            #perché unqueeze? le reti neurali vogliono le etichette in due dimensioni, nel formato: (N_labels,1)
            #clamp per evitare che ci siano delle cifre decimali sul max 
        


if __name__ == "__main__":

    #example of usage

    #first declare the dataset object
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/train')

    #then give the dataset object to the dataloader
    data_loader = TransformDataLoader(cropping_mode=RandomTransform.GLOBAL_CROP,dataset=dataset,num_workers=8,batch_size=32,dataset_mode=DirectoryRandomDataset.COUP)


    num_iterations = 0

    #now iterate like that on the dataloader
    for images, labels in data_loader:
        print(images.size())
        print(labels.size())
        num_iterations+=1

        '''for i in range(images.size(0)):
            
            path=''
            if labels[i]==DirectoryRandomDataset.FAKE:
                path = os.path.expanduser(f'~/CVCSproj/outputs/garbage/FAKE_{i}.PNG')
            elif labels[i]==DirectoryRandomDataset.REAL:
                path = os.path.expanduser(f'~/CVCSproj/outputs/garbage/REAL_{i}.PNG')
            else:
                raise RuntimeError(f'Unexpected label found: {labels[i]}')
            
            save_image(images[i],path)'''

        