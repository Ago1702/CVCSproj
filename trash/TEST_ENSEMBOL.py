import torch
from torch import nn as nn
from data.datasets import DirectorySequentialDataset
from data.datasets import DirectoryRandomDataset
import signaling.signalnet
import signaling.wavelets
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader
import models.nets as nets
from utils.helpers import load_checkpoint
import signaling

torch.cuda.manual_seed_all(42)

torch.backends.cudnn.enabled = False
while True:
    transformer = nn.DataParallel(nets.ViT_3()).cuda()
    cbam = nn.DataParallel(nets.cbam_classifier_152()).cuda()
    signal_net = nn.DataParallel(signaling.signalnet.SignalNet(do_wavelet_transform=True)).cuda()
    
    print('ViT checkpoint:' + str(load_checkpoint('vit_classifier',model=transformer,iteration_index = 10000)))#9000
    print('CBAM checkpoint:' + str(load_checkpoint('ch_cbam152_contrastive',model=cbam,iteration_index=4000))) #1000
    print('SignalNet checkpoint:' + str(load_checkpoint('wave50_classifier',model=signal_net)))
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
    transformer.eval()
    cbam.eval()
    
    wavelet =  signaling.wavelets.WaveletTransform()    
    with torch.no_grad():
        accuracy = 0.0
        max_iter = 0
        print('Dataset Iteration')
        iter_n = 0
        for test_images, test_labels in test_dataloader:
            print(iter_n,flush=True)
            with torch.no_grad():
                cbam_pred = cbam(test_images)
                transformer_pred = transformer(test_images)
                signalnet_pred = signal_net(test_images)
                test_pred = torch.round(torch.sigmoid(0.15 * cbam_pred + 0.85 * transformer_pred + 0.05 * signalnet_pred))
                max_iter += test_pred.shape[0]
                good_answers = torch.sum(test_pred == test_labels)
                accuracy+=good_answers.item()
                iter_n +=100
        
        print('Accuracy is -->' + str(accuracy*100/max_iter) + '%')
    break