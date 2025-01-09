import torch
from torch import nn as nn
from data.datasets import DirectorySequentialDataset
from data.datasets import DirectoryRandomDataset
import signaling.wavelets
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
import models.nets as nets
from utils.helpers import load_checkpoint
import signaling
import argparse
import sys
from models import nets
import signaling.signalnet as signalnet
import gdown
import os
import zipfile
import shutil
from tqdm import tqdm
import random
import numpy as np

def clean_dir(is_interrupt,args,message):
    sys.stdout.flush()
    print(message)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    list_of_matches = ['part','pth','zip']
    deletable_filenames = [filename for filename in os.listdir() if any(match in filename for match in list_of_matches)]
    if len(deletable_filenames) > 0:
        for file in deletable_filenames:
            os.remove(file)
    if is_interrupt or (not args.keep_test_dir):
        if os.path.exists('test'):
            shutil.rmtree('test')
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print('cuda not available: cringe!')
    try:
        model_choices = [
            'resnet50' , 'resnet50cbamcontrastive' , 
            'resnet152', 'resnet152cbam',
            'vit','wavenet','ensemble','code'
        ]
        parser = argparse.ArgumentParser(
            description="test script for our models"
        )

        parser.add_argument(
            '-m', '--model',
            type=str,
            help=f"Choose a model from: {', '.join(model_choices)}",
            required=True,
        )
        parser.add_argument(
            '--keep-test-dir',
            action='store_true',
            help="Keep the test directory after the script finishes."
)
        parser.add_argument(
            '--download_test_set',
            action = 'store_true',
            help = 'Download the test set (useful if testing on a local machine)'
        )
        args = parser.parse_args()
        print('Hi! You are now testing our model! Relax, it will take a while...')
        if not args.model in model_choices:
            print(f'Invalid model argument: {args.model}')
            print(f'Try one of {model_choices}')
            sys.exit(1)
        
        models_weights_dict = {
            'resnet50':                 'https://drive.google.com/uc?id=150nfmRGFLTWo8uQ8W1t6cg73sZotOpXK',
            'resnet50cbamcontrastive':  'https://drive.google.com/uc?id=1JkFiuDOkt1Wq3nZAk7dr1XyvCIKJetkG',
            'resnet152':                'https://drive.google.com/uc?id=1HL--zu1VRlUEcc0bnuBiglXMDASSe2VY',
            'resnet152cbam':            'https://drive.google.com/uc?id=1pfnvDuPJ_oAUH9TawN-P8olULDaprEb9',
            'vit':                      'https://drive.google.com/uc?id=19c2-zr7I9aAhAce5W0Tfd9MTpE9aS6_c',
            'wavenet':                  'https://drive.google.com/uc?id=1MuSdfqPd-yOR_ykk1ure7NUH3zTzGHLW', 
            'ensemble':                 'https://drive.google.com/uc?id=1dzsHe-BNYUIWzCzJN8Ktskmakh318TfL'
        }
        
        models_models_dict = {
            'resnet50':                 nets.vanilla_resnet_classifier_50(),
            'resnet50cbamcontrastive':  nets.cbam_classifier_50(),
            'resnet152':                nets.vanilla_resnet_classifier_152(),
            'resnet152cbam':            nets.cbam_classifier_152(),
            'vit':                      nets.ViT_3(),
            'wavenet':                  signalnet.SignalNet(), 
            'ensemble':                 nets.SuperEnsemble(),
            'code' :                    nets.VITContrastiveHF(classificator_type='linear')
        }
        
        # Set seed for CPU
        torch.manual_seed(42)

        # Set seed for CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)  # If multiple GPUs are being used

        # Set seed for other libraries (optional, but useful for reproducibility)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.enabled = False

        model = nn.DataParallel(models_models_dict[args.model].to(device))
        if args.model != 'code':
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            print('Downloading weights file...')
            gdown.download(
                    url=models_weights_dict[args.model]
                )

            weights_filename = [filename for filename in os.listdir() if 'pth' in filename][0]
            model.load_state_dict(torch.load(weights_filename,weights_only=False)['model'])
            os.remove(weights_filename)
            print('Weights file successfully loaded!')
        is_code = True if args.model == 'code' else False
        test_dataset = DirectorySequentialDataset(dir='/work/cvcs2024/VisionWise/test',is_code=is_code)
        if args.download_test_set:
            
            print('Downloading the dataset')
            gdown.download(
                    url='https://drive.google.com/uc?id=19nrUNb4U3PCgCDTUGFYNPlK1ZHZwI61S'
                )
            print('Extracting the dataset...')
            with zipfile.ZipFile('test.zip') as zip_ref:
                zip_ref.extractall()
            os.remove('test.zip')
            test_dataset = DirectorySequentialDataset(dir='test',is_code=is_code)
        #torch.use_deterministic_algorithms(True)
        #dataset and dataloader for testing

        test_dataloader = TransformDataLoader(
            cropping_mode=RandomTransform.GLOBAL_CROP,
            dataset=test_dataset,
            batch_size=100,
            num_workers=4,
            dataset_mode=DirectoryRandomDataset.COUP,
            probability=0.0,
            center_crop=True
            )

        model.eval()
        with torch.no_grad():
            good_answers = 0.0
            good_real_answers = 0.0
            good_fake_answers = 0.0
            max_iter = 0
            print('Dataset Iteration')
            iter_n = 0
            for test_images, test_labels in tqdm(test_dataloader,desc='Testing progress...',):
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                
                test_pred = torch.sigmoid(model(test_images))
                test_pred = torch.round(test_pred)
                max_iter += test_pred.shape[0]
                good_answers += torch.sum(test_pred == test_labels)
                good_real_answers += torch.sum(torch.logical_and(test_pred == 0, test_labels == 0))
                good_fake_answers += torch.sum(torch.logical_and(test_pred == 1, test_labels == 1))

            
                iter_n +=100
            
            print('Accuracy is -->' + str(good_answers.item()*100/max_iter) + '%')
            print('Real accuracy is -->' + str(good_real_answers.item()*200/max_iter) + '%')
            print('Fake accuracy is -->' + str(good_fake_answers.item()*200/max_iter) + '%')
            
        clean_dir(is_interrupt=False,args=args,message='End of testing. Cleaning...')
    except KeyboardInterrupt:
        clean_dir(is_interrupt=True,args=args,message='Keyboard interrupt detected. Cleaning...')