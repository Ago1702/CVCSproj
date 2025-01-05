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
if __name__ == '__main__':
    try:
        print('Hi! You are now testing our model! Relax, it will take a while...')
        model_choices = [
            'resnet50' , 'resnet50cbamcontrastive' , 
            'resnet152', 'resnet152cbam',
            'vit','wavenet','ensemble'
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

        args = parser.parse_args()
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
            'ensemble':                 nets.SuperEnsemble()
        }
        
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.enabled = False

        model = nn.DataParallel(models_models_dict[args.model])
        
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
        print('Downloading the dataset')
        gdown.download(
                url='https://drive.google.com/uc?id=19nrUNb4U3PCgCDTUGFYNPlK1ZHZwI61S'
            )
        print('Extracting the dataset...')
        with zipfile.ZipFile('test.zip') as zip_ref:
            zip_ref.extractall()
        
        #torch.use_deterministic_algorithms(True)
        #dataset and dataloader for testing
        test_dataset = DirectorySequentialDataset(dir='/work/cvcs2024/VisionWise/test')
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
            accuracy = 0.0
            max_iter = 0
            print('Dataset Iteration')
            iter_n = 0
            for test_images, test_labels in test_dataloader:
                print(iter_n)
                
                test_pred = torch.sigmoid(model(test_images))
                test_pred = torch.round(test_pred)
                max_iter += test_pred.shape[0]
                good_answers = torch.sum(test_pred == test_labels)
                accuracy+=good_answers.item()
                iter_n +=100
            
            print('Accuracy is -->' + str(accuracy*100/max_iter) + '%')
    except KeyboardInterrupt:
        sys.stdout.flush()
        print('Keyboard interrupt detected. Cleaning...')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        list_of_matches = ['part','pth','zip']
        deletable_filenames = [filename for filename in os.listdir() if any(match in filename for match in list_of_matches)]
        if len(deletable_filenames) > 0:
            for file in deletable_filenames:
                os.remove(file)
        if os.path.exists('test'):
            os.removedirs('test')
        