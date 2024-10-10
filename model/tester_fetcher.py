import os
import random
import shutil
import argparse
import sys 
import io
import requests
import pandas as pd
import re
from PIL import Image

from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files

def extract_number(name):
    # Find the first sequence of digits in the string
    match = re.search(r'\d+', name)
    if match:
        return int(match.group())
    return None

def train_test_splitter(files: list[str]) -> tuple[list[str] , list[str]]:
    train_list: list[str] = []
    test_list: list[str] = []
    for file in files:
        if 'valid' not in file:
            train_list.append(file)
        else:
            test_list.append(file)
    return (train_list,test_list)

repo_id = "elsaEU/ELSA_D3"
all_files = list_repo_files(repo_id=repo_id,repo_type="dataset")
train_files, test_files =train_test_splitter(all_files)

for element in train_files:  
    try:
        number = extract_number(element)
        if number % 2 == 0:    
            train_files.remove(element)
    except:
        pass
print(train_files)