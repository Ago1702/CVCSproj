import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    '''
    A simple classifier that tries to classify if an image is real or fake, given a pretrained input embedding
    
    '''
    def __init__(self,input_size:int = 512,n_heads:int = 4,hidden_dim:int = 1024,num_encoder_layers:int = 2):
        '''
        Args:
            input_size (int) : the size of the input embedding
            n_heads (int) : the number of heads in the attention
            hidden_dim (int) : the size of the hidden layer of the module
            num_encoder_layers (int) : the number of encoder layers in the nn.TransformerEncoder
        '''
        super(TransformerClassifier,self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=n_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,enable_nested_tensor=False)
        self.fc = nn.Sequential(nn.Linear(in_features=input_size,out_features=1),nn.Sigmoid())

    def forward(self,x:torch.Tensor):
        x = x.unsqueeze(0)
        x = F.normalize(x,p=2,dim=1,eps=1e-6)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    #testing if the shapes match
    input = torch.zeros(10,512)
    net = TransformerClassifier(input_size=512,n_heads=4,hidden_dim=1024,num_encoder_layers=2)
    out = net(input)
    print(out.shape)
        


