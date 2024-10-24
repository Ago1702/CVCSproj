import cv2
import cv2.data
import torch
import random
import numpy as np
import math
import torch.nn.functional as F
from numpy.random import PCG64
from skimage import data
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import os

class GridRandomCrop(v2.Transform):
    """
    Class for cropping images produced by the GridTransform.

    It operates like a RandomCrop, but picks the random crop from the center area of the image, to avoid selecting
    the padding area

    """
    def __init__(self,size: tuple[int,int],cropping_window:tuple[float,float]=(7/30,13/30)):
        super().__init__()
        #checking the parameters
        if size[0] <= 0 or size[1] <= 0:
            raise(ValueError(f"Invalid size parameter:({size[0]},{size[1]}): it must be a tuple of two positive integers")) 
        self.size = size
        self.cropping_window = cropping_window
    


    def forward(self,image:torch.Tensor)->torch.Tensor:
        """
        Apply the crop on the image(s)

        Args:
            x (torch.Tensor): The input image

        Returns:
            torch.Tensor: The transformed image
        """
        y = torch.randint(low=int(image.size(2)*self.cropping_window[0]),high=int(image.size(2)*self.cropping_window[1]),size=(1,))
        x = torch.randint(low=int(image.size(3)*self.cropping_window[0]),high=int(image.size(3)*self.cropping_window[1]),size=(1,))
        return v2.functional.crop(image,y,x,self.size[0],self.size[1])
    
    def __call__(self,image:torch.Tensor)->torch.Tensor:
        return self.forward(image)

class GridTransform(v2.Transform):
    """
    Class for transforming an image into a 3x3 grid with 9 copies of the original image

    This class inherits from v2.Transform to make it compatible with the other transformation.
    """
    def __init__(self):
        super().__init__()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        Apply the conversion from image to 3x3 grid of the same image

        Args:
            x (torch.Tensor): The input image

        Returns:
            torch.Tensor: The transformed image
        """
        x_grid = x.repeat(1, 1, 3, 3)
        return x_grid
    
    def __call__(self,x:torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    
class RandomTransform(v2.Transform):
    """
    Class for increasing prediction robusteness via data agumentation

    This class applies a random sub-set of transformations to the images.
    The possible transformations include: 
    - Perspective transformation
    - Rotation
    - Blurring
    - Sharpening
    After the optional transformations, if pacman==true, the images are repeated 9 times in a 3x3 grid.
    Then, crop of the images is applied. The purpose of the crop is to provide both robustness
    and resizing of the data.

    Attributes:

        p (float): probabilty to undergo a transformation
        scale (float): probability scale factor
        transform (tuple): transformation pool
        SEED (int): seed for pseudo random processes
    """
    GLOBAL_CROP: int = 0
    LOCAL_CROP: int = 1
    GLOBAL_CROP_SIZE: tuple[int,int] = (200,200)
    LOCAL_CROP_SIZE: tuple[int,int]= (80,80)
    def __init__(self, p:float = 0.5, scale:float=0.8,
                 transform:tuple = (v2.RandomPerspective(p=1), v2.RandomAffine(90), v2.GaussianBlur(3), v2.RandomAdjustSharpness(2, 1),
                                    v2.RandomHorizontalFlip(p=1),v2.RandomVerticalFlip(p=1)),
                 SEED:int = 30347, cropping_mode:int = GLOBAL_CROP, pacman: bool = False):
        super().__init__()
    
        # checking the values of the provided parameters
        if p < 0 or p > 1:
            raise ValueError(f"Invalid value for p:{p}. p must be in the range [0, 1].")
        if scale < 0:
            raise ValueError(f"Invalid value for scale:{scale}. p must be a positive number.")    
        if cropping_mode != self.GLOBAL_CROP and cropping_mode != self.LOCAL_CROP:
            raise ValueError(f"Invalid value for cropping_mode: cropping_mode must be either GLOBAL_CROP (0) or LOCAL_CROP (1).")
        
        self.trf = transform
        self.p = p
        self.scale = scale
        self.permuter = np.random.Generator(PCG64(SEED))
        self.rnd = random.Random(SEED)
        self.cropping_mode = cropping_mode
        self.pacman = pacman
        pass

    def get_transform(self,image:torch.Tensor) -> v2.Transform:
        """
        Elaborate the subset of transforms.
        After a random permutation of the tranform set, the function start iterating over the transformations.
        For each iteration the current transformation is taken with a probability p.
        If the transformation is applied the probability is scaled by scale.
        Otherwise break.

        Returns:
            v2.Transform: the ordered composition of the taken transformation
        """        
        perm = self.permuter.permutation(x = len(self.trf))
        p = self.p
        trfs = list()
        if self.pacman:
            trfs.append(GridTransform())
        for i in perm:
            if p > self.rnd.random():
                trfs.append(self.trf[i])
                p *= self.scale
                p = min(max(0, p), 1)
            else:
                break
        size:tuple = (0,0)
        if self.cropping_mode==self.GLOBAL_CROP:
            size = self.GLOBAL_CROP_SIZE
        if self.cropping_mode==self.LOCAL_CROP:
            size = self.LOCAL_CROP_SIZE
        
        if self.pacman:
            if image.size(2)>350 and self.cropping_mode==self.LOCAL_CROP:
                trfs.append(GridRandomCrop(size,(6/30,14/30)))
            else:
                trfs.append(GridRandomCrop(size))
        else:
            trfs.append(v2.RandomCrop(size=size))
        comp = v2.Compose(trfs)
        return comp

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Apply a random subset of transformation to an image.
        For more details about the random set view 'get_transform' method.

        Args:
            x (torch.Tensor): The input image

        Returns:
            torch.Tensor: The transformed image
        """
        if x.dim() != 4:
            raise RuntimeError('Tensors must be 4 dimensional: batch_size, channels, height, width') 
        smallest_dimension = min(x.size(2),x.size(3))
        if smallest_dimension < 200:
            scale = math.ceil(200/smallest_dimension)
            if scale > 20:
                raise RuntimeError(f'Strange value for scale: {scale}. Dimensions were {x.size(0)} , {x.size(1)},{x.size(2)} , {x.size(3)}')
            
            x = F.interpolate(x,scale_factor=scale, mode='bilinear', align_corners=False)
            
        return self.get_transform(x).forward(x)
    
    def __call__(self, x)->torch.Tensor:
        return self.forward(x)
    
#   Test Code
'''
if __name__ == "__main__":
    N = 10 #N is the number of different transformed images that will be generated by the transformations
    
    #getting the image from skimage,
    caller = getattr(data, 'astronaut')
    image = caller()
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    torch_converter = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    input_image = torch_converter(image)

    try:
        os.remove('outputs/input_image.jpg')
    except:
        pass
    save_image(input_image,'outputs/input_image.jpg')

    grid_transform=GridTransform()
    grid_image=grid_transform(input_image)
    save_image(grid_image,'outputs/grid_image.jpg')
    print(grid_image.size())
    for i in range(N):
        random_transform = RandomTransform(p=1, scale=1,cropping_mode=RandomTransform.LOCAL_CROP,pacman=True)
        output_image=random_transform(input_image)
        image_filename = 'outputs/output_image'+ str(i)+'.jpg'
        try:
            os.remove(image_filename)
        except:
            pass
        save_image(output_image,image_filename)
    pass
'''

