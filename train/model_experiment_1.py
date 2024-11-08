import sys
import os
sys.path.append(os.path.expanduser('/homes/ocarpentiero/CVCSproj/model/'))

import torch
import torch.nn as nn
import torch.optim as optim
from utils.modules import MultiCBAM
from data.datasets import DirectoryRandomDataset
from utils.transform import RandomTransform
from data.dataloader import TransformDataLoader


class DummyCBAM_1(nn.Module):
    def __init__(self):
        super(DummyCBAM_1, self).__init__()
        # defining layers
        self.net = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1), #3x200x200 --> 32x200x200
        nn.MaxPool2d(kernel_size=2,stride=1), #32x200x200 --> 32x199x199
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1), #32x199x199 --> 64x199x199
        nn.MaxPool2d(kernel_size=2,stride=1,padding=1), #64x199x199 --> 64x200x200
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1), #64x200x200 --> 128x200x200
        nn.MaxPool2d(kernel_size=2,stride=1,padding=1), #128x200x200 --> 128x201x201
        nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1), #128x201x201 --> 64x201x201
        nn.MaxPool2d(kernel_size=3,stride=2), #64x101x101
        nn.Conv2d(in_channels=64,out_channels=16,kernel_size=3,stride=1,padding=1), #64x101x101 --> 16x101x101
        nn.MaxPool2d(kernel_size=3,stride=2), #16x101x101 --> 16x51x51
        nn.Conv2d(in_channels=16,out_channels=4,kernel_size=3,stride=1), #16x51x51 --> 4x49x49
        nn.MaxPool2d(kernel_size=3,stride=2), #4x49x49 --> 4x25x25
        nn.Conv2d(in_channels=4,out_channels=1,kernel_size=3,stride=1), #4x25x25 --> 1x23x23
        nn.MaxPool2d(kernel_size=2,stride=1), #1x21x21 --> 1x20x20
        nn.Flatten(),
        )
        self.fully_connected = nn.Sequential(
        nn.Linear(400,100),
        nn.ReLU(),
        nn.Linear(100,1),
        nn.Sigmoid()
        )
    
        
    def forward(self,x:torch.Tensor):
        conv=self.net.forward(x)
        return self.fully_connected.forward(conv)
    
if __name__ == '__main__':
    torch.backends.cudnn.enabled = False

    # checking if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Did you use the slurm force?")
    dataset = DirectoryRandomDataset('/work/cvcs2024/VisionWise/test')
    dataloader = TransformDataLoader(RandomTransform.GLOBAL_CROP,dataset,num_workers=1,batch_size=32,probability=0.4,pacman=True)
    model = DummyCBAM_1().cuda()

    #let's make it parallel
    model = nn.DataParallel(model)

    num_params = sum(p.numel() for p in model.parameters())
    param = next(model.parameters())
    param_size_in_bytes = num_params * param.element_size() 
    print(param_size_in_bytes)

    #definition of the loss and the gradient descent thing
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    #model in training moden
    model.train(True)

    #loading the weights
    if os.path.exists(os.path.expanduser('~/CVCSproj/model/weights/DummyCBAM_1.pth')):
        model.load_state_dict(torch.load(os.path.expanduser('~/CVCSproj/model/weights/DummyCBAM_1.pth')))
        print('weights loaded')

    optimizer.zero_grad()
    
    batch_size = 500
    
    for index, (images, labels) in enumerate(dataloader):
        if (index+1) % 20 ==0:
            print(f'iteration number: {index + 1}',flush=True)
            
        '''print(f'max = {torch.max(images)}')
        print(f'min = {torch.min(images)}')
        print(f'avg = {torch.mean(images)}')'''
        #forward pass
        outputs = model(images)
        
        #calcolo della loss (conversione a float32 per renderlo compatibile con la cross entropy)
        loss = criterion(outputs, labels.to(torch.float32))
        loss.backward() 

        #ottimizzazione
        if (index + 1)%batch_size == 0:
            optimizer.step()
            print(f'Batch {index + 1}: Loss: {loss.item():.4f}',flush=True)  # Print the loss
            optimizer.zero_grad()
            torch.save(model.state_dict(), os.path.expanduser('~/CVCSproj/model/weights/DummyCBAM_1.pth'))
        