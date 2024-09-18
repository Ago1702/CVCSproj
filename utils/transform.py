import cv2
import cv2.data
import torch
import random
import numpy as np
from numpy.random import PCG64
from skimage import data
from torchvision import transforms as tf
from matplotlib import pyplot as plt

class Transform():
    def __init__(self, p:float = 0.5, scale:float=0.8,
                 transform:tuple = (tf.RandomPerspective(p=1), tf.RandomAffine(90), tf.GaussianBlur(3), tf.RandomAdjustSharpness(2, 1)),
                 SEED:int = 30347):
        self.trf = transform
        self.p = p
        self.scale = scale
        self.permuter = np.random.Generator(PCG64(SEED))
        self.rnd = random.Random(SEED)
        pass

    def apply_transform(self, x:torch.Tensor) -> torch.Tensor:
        perm = self.permuter.permutation(x = len(self.trf))
        p = self.p
        for i in perm:
            if p > self.rnd.random():
                x = self.trf[i](x)
                p *= self.scale
        return x

if __name__ == "__main__":
    img = torch.from_numpy(data.camera()).unsqueeze(0)
    trf = Transform(p=1, scale=1)
    img = trf.apply_transform(img)
    plt.imshow(img.squeeze())
    plt.show()
    pass