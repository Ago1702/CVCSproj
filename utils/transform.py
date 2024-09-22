import cv2
import cv2.data
import torch
import random
import numpy as np
from numpy.random import PCG64
from skimage import data
from torchvision.transforms import v2
from matplotlib import pyplot as plt

class Transform():
    def __init__(self, p:float = 0.5, scale:float=0.8,
                 transform:tuple = (v2.RandomPerspective(p=1), v2.RandomAffine(90), v2.GaussianBlur(3), v2.RandomAdjustSharpness(2, 1)),
                 SEED:int = 30347):
        
        if p < 0 or p > 1:
            raise ValueError("p in range [0, 1]")
        if scale < 0:
            raise ValueError("Solo valori positivi")
        
        self.trf = transform
        self.p = p
        self.scale = scale
        self.permuter = np.random.Generator(PCG64(SEED))
        self.rnd = random.Random(SEED)
        pass

    def get_transform(self) -> v2.Transform:
        perm = self.permuter.permutation(x = len(self.trf))
        p = self.p
        trfs = list()
        for i in perm:
            if p > self.rnd.random():
                trfs.append(self.trf[i])
                p *= self.scale
            else:
                break
        comp = v2.Compose(trfs)
        return comp

    def apply_transform(self, x:torch.Tensor) -> torch.Tensor:
        comp = self.get_transform()
        return comp.forward(x)

if __name__ == "__main__":
    N = 3
    img = torch.rand((N, 1, 150, 150))
    trf = Transform(p=1, scale=1)
    img = trf.apply_transform(img)
    for i in range(N):
        plt.imshow(img[i].squeeze())
        plt.show()
    pass
