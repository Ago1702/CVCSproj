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

def clean_directory(dir_abs_path: str):
    if dir_abs_path[0] != '/':
        raise Exception('First parameter of clean_directory must be an absolute path')

    n_of_files:int = 0
    try:
        files = os.listdir(dir_abs_path)
        for file in files:
            file_path = os.path.join(dir_abs_path,file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) 
            n_of_files = n_of_files + 1
        print(f"Successfully deleted {n_of_files} files from " + dir_abs_path)
    except:
        print("Error while deleting files from " + dir_abs_path)

def train_test_splitter(files: list[str]) -> tuple[list[str] , list[str]]:
    train_list: list[str] = []
    test_list: list[str] = []
    for file in files:
        if 'valid' not in file:
            train_list.append(file)
        else:
            test_list.append(file)
    return (train_list,test_list)

def download_file(file:str):
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type="dataset",
        local_dir=image_folder
    )

def parquet_processor(file:str,start_index:int,validation:bool = False) -> int:
    '''
    Make the parquet file suitable for our training

    This function randomly removes 3 out of 4 fake images in the parquet file (metadata and pixel data), then it adds additional columns for the corresponding real image.

    Args:
        file (str): the absolute path to the parquet file
        start_index (int): the starting index (inclusive) of the png files

    Returns:
        the next index for the following parquet_processor call
    '''
    try:
        df = pd.read_parquet(file)
    except Exception as e:
        print(str(e))

    '''
    ORIGINAL COLUMNS OF df 
    ['id', 'original_prompt', 'positive_prompt', 'negative_prompt', 'url',
       'model_gen0', 'model_gen1', 'model_gen2', 'model_gen3', 'width_gen0',
       'width_gen1', 'width_gen2', 'width_gen3', 'height_gen0', 'height_gen1',
       'height_gen2', 'height_gen3', 'num_inference_steps_gen0',
       'num_inference_steps_gen1', 'num_inference_steps_gen2',
       'num_inference_steps_gen3', 'filepath_gen0', 'filepath_gen1',
       'filepath_gen2', 'filepath_gen3', 'image_gen0', 'image_gen1',
       'image_gen2', 'image_gen3']
    '''
    index :int = start_index
    for i,row in df.iterrows():
        chosen_image=random.randint(0,3)
        image_dict = row[f'image_gen{chosen_image}']
        image_width = row[f'width_gen{chosen_image}']
        image_height = row[f'height_gen{chosen_image}']
        image_bytes = image_dict['bytes']
        image_fake = Image.open(io.BytesIO(image_bytes))
        image_code = str(index).zfill(8)

        if not validation:
            output_path_fake=f'/work/cvcs2024/VisionWise/train/train-fake-{image_code}.png'
            output_path_real=f'/work/cvcs2024/VisionWise/train/train-real-{image_code}.png'
        else:
            output_path_fake=f'/work/cvcs2024/VisionWise/test/test-fake-{image_code}.png'
            output_path_real=f'/work/cvcs2024/VisionWise/test/test-real-{image_code}.png'
        #saving fake image
        image_fake.save(output_path_fake, format='PNG')

        try:
            r = requests.get(row['url'], stream=True, timeout=5)
            if r.status_code == 200:
                image_real = Image.open(io.BytesIO(r.content))
                if image_real.mode == 'CMYK':
                    image_real = image_real.convert('RGB')
                image_real.save(output_path_real, format='PNG')
                index+=1
            else:
                os.remove(output_path_fake)
        except:
            os.remove(output_path_fake)
    os.remove(file)
    return index
        
    

#checking the args
if len(sys.argv)>2 or len(sys.argv)<2:
    print("Wrong number of parameters")
    sys.exit(1)

BLOCK_ID = int(sys.argv[1])

image_folder: str = '/work/cvcs2024/VisionWise/parquet' 


'''directories_to_clean: list[str] = ['/work/cvcs2024/VisionWise/parquet/data', '/work/cvcs2024/VisionWise/train','/work/cvcs2024/VisionWise/test']

for dir in directories_to_clean:
    clean_directory(dir)'''

#getting data from huggingface hub
repo_id = "elsaEU/ELSA_D3"
all_files = list_repo_files(repo_id=repo_id,repo_type="dataset")
train_files, test_files =train_test_splitter(all_files)

#test dataset part
'''futures = []
with ThreadPoolExecutor(max_workers=11) as executor:
        for file in test_files:
            futures.append(executor.submit(download_file,file))
            #if there is no file left to process, break
            if len(train_files)==0:
                break
    #ensuring all threads complete their execution
for future in futures:
        try:
            result = future.result()  # Wait for the thread to complete and get the result
        except Exception as e:
            print(f'Error')

print(f"Saved a total of {len(test_files)} test files")

selected_files = os.listdir(os.path.join(image_folder,'data'))
file_index:int = 0
os.system('clear')


futures = []
with ThreadPoolExecutor(max_workers=11) as executor:
    for file in selected_files:
        futures.append(executor.submit(parquet_processor,os.path.join(image_folder,'data',file),file_index,True))
        file_index+=600
        
#ensuring all threads complete their execution
for future in futures:
        try:
            result = future.result()  # Wait for the thread to complete and get the result
        except Exception as e:
            print(f'Error')'''

'''old code
for file in selected_files:
    print(f'Start Index={file_index}')
    
    file_index=parquet_processor(os.path.join(image_folder,'data',file),file_index,validation=True)
    os.remove(os.path.join(image_folder,'data',file))
    print('Parquet file ended')
    print(f'End Index={file_index}')'''


#train dataset part
file_counter:int = 0
file_index:int = BLOCK_ID * 10000
#now let's get some parquet files, one batch at a time
batch_size: int = 10

#now i select only part of the parquet files
new_list = []
for element in train_files:  
    try:
        number = extract_number(element)
        if number % 1024 == BLOCK_ID:    
            new_list.append(element)
    except:
        if BLOCK_ID == 1023:
            new_list.append(element)
        else:
            pass

train_files = new_list
selected_files = []
while(True):
    for i in range(batch_size):
            file = random.choice(train_files)
            train_files.remove(file)
            download_file(file)
            selected_files.append(file)
            file_counter +=1

            #if there is no file left to process, break
            if len(train_files)==0:
                break


    futures = []
    '''with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(batch_size):
            file = random.choice(train_files)
            train_files.remove(file)
            futures.append(executor.submit(download_file,file))
            file_counter +=1

            #if there is no file left to process, break
            if len(train_files)==0:
                break
    
#ensuring all threads complete their execution
    for future in futures:
        try:
            result = future.result()  # Wait for the thread to complete and get the result
        except Exception as e:
            print(f'Error')'''

    print(f"Saved a total of {file_counter} train files")

    #selected_files = os.listdir(os.path.join(image_folder,'data'))
    #os.system('clear')

    futures=[]
    with ThreadPoolExecutor(max_workers=16) as executor:
        for file in selected_files:
            futures.append(executor.submit(parquet_processor,os.path.join(image_folder,file),file_index))
            file_index+=600
    
    #ensuring all threads complete their execution
    for future in futures:
        try:
            result = future.result()  # Wait for the thread to complete and get the result
        except Exception as e:
            print(f'Error')

    '''old code
    for file in selected_files:
        print(f'Start Index={file_index}')
        file_index=parquet_processor(os.path.join(image_folder,'data',file),file_index)
        os.remove(os.path.join(image_folder,'data',file))
        print('Parquet file ended')
        print(f'End Index={file_index}')
'''
    if len(train_files)==0:
        break
