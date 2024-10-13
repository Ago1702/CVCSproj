import torch
import random
from typing import Union
import os
import io
import re
import subprocess
import numpy as np
import requests
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
import torchvision.transforms.v2 as transforms
#from datasets import load_dataset
from torchvision.io import read_image
from PIL import Image
#import matplotlib.pyplot as plt

class BufferedIterable(IterableDataset):
    """
    A wrapper for Iterable Dataset, implements a buffer to simulate a local shuffling

    suggested by:
        sharvil Sharvil Nanavati: https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/4

    Args:
            buffer_size (int)
    """    

    FAKE:int = 1
    REAL:int = 0

    def __init__(self, buffer_size:int):
        super(BufferedIterable).__init__()
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.dataset = []#load_dataset("elsaEU/ELSA1M_track1", split="train", streaming=True)
        self.buffer_size = buffer_size
    
    def __iter__(self):
        buff = []
        
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                cat = np.random.choice([self.REAL, self.FAKE])
                item = next(dataset_iter)
                img = item["image"]
                if(cat == self.REAL):
                    r = requests.get(item["url_real_image"], stream=True)
                    if r.status_code == 200:
                        img = Image.open(io.BytesIO(r.content))
                    else:
                        cat = self.FAKE
                img = self.__transform_image__(img)
                buff.append((img, cat))
        except:
            self.buffer_size = len(buff)
        
        try:
            while True:
                try:
                    item = next(dataset_iter)
                    cat = np.random.choice([self.REAL, self.FAKE])
                    img = item["image"]
                    if(cat == self.REAL):
                        r = requests.get(item["url_real_image"], stream=True)
                        if r.status_code == 200:
                            img = Image.open(io.BytesIO(r.content))
                        else:
                            cat = self.FAKE
                    img = self.__transform_image__(img)
                    rem_idx = random.randint(0, self.buffer_size - 1)
                    yield buff[rem_idx]
                    buff[rem_idx] = (img, cat)
                except StopIteration:
                    break
            while len(buff) > 0:
                yield buff.pop()
        except GeneratorExit:
            pass

    def __transform_image__(self, img:Image) -> torch.Tensor:
        """
        A simple transform. This change to image to a format visualizable with imshow from matplotilb

        Args:
            img (Image)

        Returns:
            torch.Tensor
        """        
        tnr = self.transform(img)
        tnr = torch.swapaxes(tnr, 0, 2)
        tnr = torch.swapaxes(tnr, 0, 1)
        return tnr


class DirectoryRandomDataset(IterableDataset):
    FAKE:int = 1
    REAL:int = 0

    def __init__(self, dir: Union[str, Path], ext:str = "png"):
        super().__init__()
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("The path have to be a directory")
        self.dir = dir
        out = subprocess.check_output(f"ls {self.dir.resolve()} | tail -n 1", shell=True)
        self.leng = int(re.findall(r"\d+", str(out))[0])
        self.label = {0 : "real", 1 : "fake"}
        self.ext = ext
        self.tensorizzatore = transforms.Compose([transforms.PILToTensor()])


    def __iter__(self):
        while(True):
            i = np.random.choice(self.leng)
            rf = np.random.choice([0, 1])
            p = self.dir / f"test-{self.label[rf]}-{i:08d}.{self.ext}"
            if p.exists():
                image = Image.open(p).convert("RGB")
                return self.tensorizzatore(image), torch.tensor(rf, dtype=torch.long)


#   Test code
"""
if __name__ == "__main__":
    ds = BufferedIterable(4)
    t = transforms.PILToTensor()
    i = 10
    for el in ds:
        if(i == 0):
            break
        plt.imshow(el[0])
        plt.show()
        print(el[1])
        i-=1
"""
if __name__ == "__main__":
    ds = DirectoryRandomDataset("//work//cvcs2024//VisionWise//test")
    print(ds.__iter__())