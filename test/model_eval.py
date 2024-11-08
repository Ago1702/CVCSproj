import torch
import torch.nn as nn
from data.datasets import DirectorySequentialDataset
from torch.utils.data import DataLoader
from utils.transform import RandomTransform
from utils.helpers import point_module_remover
from models.resnet_cbam import v2

if __name__ == '__main__':
    dataset = DirectorySequentialDataset(dir='/work/cvcs2024/VisionWise/test')
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=False)
    transform = RandomTransform()
    
    embedder = v2()
    classifier = nn.Sequential(nn.Linear(in_features=512,out_features=1))
    
    model = nn.Sequential(embedder,classifier)
    
    state_dict = torch.load('/work/cvcs2024/VisionWise/weights/linear_classifier_v0_18500.pth',weights_only=True)
    state_dict = point_module_remover(state_dict=state_dict)
    
    model.load_state_dict(state_dict=state_dict)
    model = nn.Sequential(model,nn.Sigmoid())
    model=model.cuda()
    model.eval()
    
    num_of_expantions = 50 #number of times the same image is transformed
    '''
    idea:   we only consider a small portion of an image. feeding it through the network (cropping it differently) multiple times COULD give
            better results
    '''
    num_of_forwards = 0
    good_results = 0
    for n, (image_r,image_f) in enumerate(dataloader):
        image_r_expanded_list = []
        image_f_expanded_list = []
        for i in range(num_of_expantions):
            image_r_expanded_list.append(transform(image_r).cuda())
            image_f_expanded_list.append(transform(image_f).cuda())
            
        images_r = torch.concat(image_r_expanded_list).cuda()
        images_f = torch.concat(image_f_expanded_list).cuda()
        
        result_r = model(images_r).sum()
        result_f = model(images_f).sum()
        print(result_r)
        print(result_f)
        if result_r < 25:
            good_results +=1
            
        if result_f > 25:
            good_results +=1
            
        num_of_forwards += 2
        
        print(f'current accuracy: {good_results/num_of_forwards}')
        