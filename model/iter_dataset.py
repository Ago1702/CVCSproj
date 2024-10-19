import torch
import random
import json
import os
import io
import re
import subprocess
import numpy as np
import requests
from abc import ABC, abstractmethod
from typing import Union
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
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

"""class DirectoryDataset(ABC):
    def __init__(self, dir: Union[str, Path], ext:str = "png"):
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("The path have to be a directory")
        self.dir = dir
        self.ext = ext
        if not (self.dir / ".info.json").exists():
            self.__write_info__()
        else:


    def __write_info__()"""

class DirectoryRandomDataset(IterableDataset):
    """
    An Iterable dataset for handling the enormous datatset of ELSA_D3

    There are 4 main iteration modes:
        - Extract a random image
        - Extract a random real image
        - Extract a random fake image
        - Extract a random couple real/fake (semantic correlation)
    
    Use change_mod to change de modality
        - BASE for 1st mode
        - REAL for 2nd mode
        - FAKE for 3rd mode
        - COUP for 4th mode
    - 
    """

    FAKE:int = 1
    REAL:int = 0
    BASE:int = 2
    COUP:int = 3

    def __init__(self, dir: Union[str, Path], max_iter:int = 0, ext:str = "png", check:bool = False):
        super().__init__()
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("The path have to be a directory")
        self.dir = dir
        if not (self.dir / ".info.json").exists() or check:
            self.__write_info__()
        else:
            with open(self.dir / ".info.json") as f:
                info = json.load(f)
                self.len = info["len"]
        self.len = self.len // 2
        self.label = {0 : "real", 1 : "fake"}
        self.ext = ext
        self.tensorizzatore = transforms.Compose([transforms.PILToTensor()])
        self.behaviour = self.__random_image__
        self.max_iter = max_iter
        self.iter = 0
    
    def __write_info__(self):
        out = subprocess.check_output(f"ls -1 {self.dir.resolve()} | wc -l", shell=True)
        self.len = int(re.findall(r"\d+", str(out))[0])
        data = {"len": self.len}
        with open(self.dir / ".info.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def __iter__(self):
        while(self.iter < self.max_iter or self.max_iter == 0):
            i = np.random.choice(self.len)
            yield self.behaviour(i)
            self.iter += 1
        raise StopIteration

    def __random_real__(self, i:int):
        p = self.dir / f"image-real-{i:08d}.{self.ext}"
        if p.exists():
            image = Image.open(p).convert("RGB")
            return self.tensorizzatore(image).unsqueeze(0).type(torch.float32), torch.tensor(0, dtype=torch.long)
        else:
            raise RuntimeError("Image not present, some problem occur")

    def __random_fake__(self, i:int):
        p = self.dir / f"image-fake-{i:08d}.{self.ext}"
        if p.exists():
            image = Image.open(p).convert("RGB")
            return self.tensorizzatore(image).unsqueeze(0).type(torch.float32), torch.tensor(1, dtype=torch.long)
        raise RuntimeError("Image not present, some problem occur")
    
    def __random_couple__(self, i:int):
        pr = self.dir / f"image-real-{i:08d}.{self.ext}"
        pf = self.dir / f"image-fake-{i:08d}.{self.ext}"
        if pf.exists() and pr.exists():
            imagef = Image.open(pf).convert("RGB")
            imager = Image.open(pr).convert("RGB")
            return self.tensorizzatore(imager).unsqueeze(0).type(torch.float32), self.tensorizzatore(imagef).unsqueeze(0).type(torch.float32)
        raise RuntimeError("Image not present, some problem occur")
    
    def __random_image__(self, i:int):
        rf = np.random.choice([0, 1])
        p = self.dir / f"image-{self.label[rf]}-{i:08d}.{self.ext}"
        if p.exists():
            image = Image.open(p).convert("RGB")
            return self.tensorizzatore(image).unsqueeze(0).type(torch.float32), torch.tensor(rf, dtype=torch.long)
        raise RuntimeError("Image not present, some problem occur")
    
    def change_mode(self, mode:int):
        if mode == self.BASE:
            self.behaviour = self.__random_image__
        elif mode == self.REAL:
            self.behaviour = self.__random_real__
        elif mode == self.FAKE:
            self.behaviour = self.__random_fake__
        elif mode == self.COUP:
            self.behaviour = self.__random_couple__

class DirectorySequentialDataset(Dataset):
    def __init__(self, dir: Union[str, Path], ext:str = "png"):
        super().__init__()
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("The path have to be a directory")
        self.dir = dir
        self.ext = ext
        self.tensorizzatore = transforms.Compose([transforms.PILToTensor()])
        self.label = {0:"real", 1:"fake"}
        if not (self.dir / ".info").exists():
            self.__write_info__()
        else:
            with open(self.dir / ".info", "r") as f:
                try:
                    self.len = int(f.readline())
                except Exception:
                    self.__write_info__()
            if (self.dir / f"image-real-{self.len//2:08d}.{self.ext}").exists():
                self.__write_info__()

    def __write_info__(self):
        out = subprocess.check_output(f"ls -1 {self.dir.resolve()} | wc -l", shell=True)
        self.len = int(re.findall(r"\d+", str(out))[0])
        if(self.len % 2 != 0):
            raise Exception("Someting wrong in enumeration")

        with open(self.dir / ".info", "w") as f:
            f.write(str(self.len))
            f.close()

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if index >= self.len or index < 0:
            raise IndexError()
        return self.tensorizzatore(Image.open(self.dir / f"image-{self.label[index % 2]}-{index//2:08d}.{self.ext}").convert("RGB")), torch.tensor(index % 2, dtype=torch.long)

"""
TODO
    Migliorare il dataset sequenziale
TODO
    Creare una superclasse per la gestione info
"""

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

if __name__ == "__main__":
    ds = DirectoryRandomDataset(100, "//work//cvcs2024//VisionWise//test")
    ds.change_mode(DirectoryRandomDataset.COUP)
    it = ds.__iter__()
    for i in range(10):
        img = next(it)
        save_image(img[0].double()/255, f"test_image/image{i}-0.png")
        save_image(img[1].double()/255, f"test_image/image{i}-1.png")
    print(ds.iter)

"""