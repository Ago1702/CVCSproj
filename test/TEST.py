import torch
from torch import nn as nn
from data.datasets import DirectorySequentialDataset
from data.datasets import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
import models.nets as nets
from utils.helpers import load_checkpoint

torch.cuda.manual_seed_all(42)
torch.backends.cudnn.enabled = False
while True:
    model = nn.DataParallel(nets.cbam_classifier_152()).cuda()
    #model.load_state_dict(torch.load('/work/cvcs2024/VisionWise/weights/ch_cbam152_classifier_2000.pth',weights_only=False)['model'])
    print('loaded checkpoint:' + str(load_checkpoint('ch_cbam152_contrastive_classifier',model=model)))
    
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
        for test_images, test_labels in test_dataloader:
            with torch.no_grad():
                test_pred = torch.sigmoid(model(test_images))
                test_pred = torch.round(test_pred)
                max_iter += test_pred.shape[0]
                good_answers = torch.sum(test_pred == test_labels)
                accuracy+=good_answers.item()
        
        print('Accuracy is -->' + str(accuracy*100/max_iter) + '%')
    break