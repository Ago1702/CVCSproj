import torch
import random
import json
import os
import io
import re
import shutil
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
from utils.transform import RandomTransform

def remove_transparency(image):
    if image.mode == 'P':
        #create a new image with a black background
        background = Image.new("RGB", image.size, (0, 0, 0))  #black background
        #paste the original image onto the background using its alpha channel as mask
        background.paste(image.convert("RGBA"), (0, 0), image.convert("RGBA"))
        return background
    return image

def to_RGB(image):
    if image.mode != 'RGB':
        return image.convert("RGB")
    return image

def is_corrupted_1x1(image):
    width, height = image.size
    if width ==1 or height ==1:
        return True
    return False

def is_damaging(image):
    width, height = image.size
    if width <100 or height <100:
        return True
    if width >5000 or height >5000:
        print(f'Found crazy image: {width}x{height}')
        return True
    return False

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
        """
        Args:
            dir (Union[str, Path]): dataset directory
            max_iter (int, optional): max iteration number, if 0 iteration are infinite. Defaults to 0.
            ext (str, optional): The image file extension. Defaults to "png".
            check (bool, optional): If true forces the creation of a .info.json file.ù
                This file wille contain dataset info. Defaults to False.
        """
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
        self.tensorizzatore = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
        self.behaviour = self.__random_couple__
        self.max_iter = max_iter
        self.iter = 0
    
    def __write_info__(self):
        """
        A method for writing the info file.
        For now it contain:
            len: the dataset size
        """
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
    
    """
    Different method for extractions.
    Check change mode
    """
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
        '''
        This behaviour extracts a random couple of semantically corresponding real and fake images.

        Returns:
            tuple[torch.Tensor, torch.Tensor] : a tuple with two tensors; the real image first, then the fake one

        '''
        while(True):
            if i<0:
                    raise RuntimeError('Negative i (????????????????????????)')
            
            pr = self.dir / f"image-real-{i:08d}.{self.ext}"
            pf = self.dir / f"image-fake-{i:08d}.{self.ext}"
            try:
                #opening the two images
                imager = Image.open(pr)
                imagef = Image.open(pf)

                #removing transparency
                imager = remove_transparency(imager)
                imagef = remove_transparency(imagef)

                #converting back to RGB 
                imager = to_RGB(imager)
                imagef = to_RGB(imagef)

                if is_corrupted_1x1(imager) or is_corrupted_1x1(imagef):
                    i-=1
                    #print('A 1x1 image was found')
                    continue
                if is_damaging(imager) or is_damaging(imagef):
                    i-=1
                    #print('A damaging image was found')
                    continue
            except:
                i-=1
                #print('A corrupted image was found')
                continue

            try:
                return self.tensorizzatore(imager).unsqueeze(0).type(torch.float32), self.tensorizzatore(imagef).unsqueeze(0).type(torch.float32)
            except:
                i-=1
                #print('A corrupted image was found')
                continue
            
    
    def __random_image__(self, i:int):
        #this part has experimental edits. porting into the other modes will possibly come into the future

        #iteration is performed to have infinite retries, if something goes wrong
        while True:
            #preparing the image paths, and the paths of the previous images
            if i<0:
                raise RuntimeError('Negative i (????????????????????????)')
            rf = np.random.choice([0, 1])
            p = self.dir / f"image-{self.label[rf]}-{i:08d}.{self.ext}"
            emergency_p = self.dir / f"image-{self.label[rf]}-{(i-1):08d}.{self.ext}"

            counterpart_rf = 1 - rf
            counterpart_p = self.dir / f"image-{self.label[counterpart_rf]}-{i:08d}.{self.ext}"
            counterpart_emergency_p = self.dir / f"image-{self.label[counterpart_rf]}-{(i-1):08d}.{self.ext}"

            #now RGBA to RGB conversion will be performed
            if p.exists():
                try:
                    image = Image.open(p)
                    #checking if the image has transparency
                    image = remove_transparency(image)

                    #converting to RGB if it's not already in that mode
                    image = to_RGB(image)

                    #checking for corrupted images
                    if is_corrupted_1x1(image):
                        '''shutil.copy(emergency_p, p)
                        shutil.copy(counterpart_emergency_p, counterpart_p)'''
                        i-=1
                        print('A 1x1 image was found')
                        continue
                except:
                    '''shutil.copy(emergency_p, p)
                    shutil.copy(counterpart_emergency_p, counterpart_p)'''
                    i-=1
                    print('A corrupted image was found')
                    continue
                
            try:    
                return self.tensorizzatore(image).unsqueeze(0).type(torch.float32), torch.tensor(rf, dtype=torch.long)
            except:
                '''shutil.copy(emergency_p, p)
                shutil.copy(counterpart_emergency_p, counterpart_p)'''
                i-=1
                print('A corrupted image was found')
                continue

    
    def change_mode(self, mode:int):
        """
        Modify the behaviour
        Use classes costant

        BASE: return a random image and label
        REAL: return a random real image and label
        FAKE: return a random fake image and label
        COUP: return two semantic correlated images the first is real, the latter is fake
        
        Args:
            mode (int): _description_
        """
        if mode == self.BASE:
            self.behaviour = self.__random_image__
        elif mode == self.REAL:
            self.behaviour = self.__random_real__
        elif mode == self.FAKE:
            self.behaviour = self.__random_fake__
        elif mode == self.COUP:
            self.behaviour = self.__random_couple__

class DirectorySequentialDataset(Dataset):
    '''
    Usage: this class is good, but do not use it directly. Pass it to a torch.utils.data.Dataloader.
    Use case: use it for the test dataset only. Sequentiality is not good for learning.
    '''
    def __init__(self, dir: Union[str, Path], ext:str = "png"):
        super().__init__()
        if not isinstance(dir, Path):
            dir = Path(dir)
        if not dir.is_dir():
            raise ValueError("The path have to be a directory")
        self.dir = dir
        self.ext = ext
        self.tensorizzatore = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
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
        if(index>=4250):
            raise StopIteration
        pr = self.dir / f"image-real-{index:08d}.{self.ext}"
        pf = self.dir / f"image-fake-{index:08d}.{self.ext}"
            
        #opening the two images
        imager = Image.open(pr)
        imagef = Image.open(pf)

        #removing transparency
        imager = remove_transparency(imager)
        imagef = remove_transparency(imagef)

        #converting back to RGB 
        imager = to_RGB(imager)
        imagef = to_RGB(imagef)

        '''if is_corrupted_1x1(imager) or is_corrupted_1x1(imagef):
            index+=1
            #print('A 1x1 image was found')
            continue
        if is_damaging(imager) or is_damaging(imagef):
            index+=1
            #print('A damaging image was found')
            continue'''
        

        return self.tensorizzatore(imager).unsqueeze(0).type(torch.float32), self.tensorizzatore(imagef).unsqueeze(0).type(torch.float32)
            
            
                
        

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