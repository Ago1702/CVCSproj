import torch
import torch.nn as nn
import os
import re
import numpy as np
from PIL import Image

def save_tensor_to_png(image_name:str,tensor:torch.Tensor,destination_dir:str = os.path.expanduser('~/CVCSproj/outputs/garbage')):
    if not image_name.endswith('.png'):
        image_name+='.png'
    save_path = os.path.join(destination_dir,image_name)
    
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(array)
    image.save(save_path)
    
def point_module_remover(state_dict):
    '''
    The weights for the model were saved when it was wrapped by a nn.DataParallel.
    That means that the keys have an extra "module." part at the beginning.
    Use this function to remove it, allowing you to load the weights in a non-parallel network
    '''
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") 
        new_state_dict[new_key] = value
        
    return new_state_dict

def state_dict_adapter(state_dict,string_to_remove:str,string_to_insert:str=''):
    '''
    The weights for the model were saved when it was wrapped by a nn.DataParallel.
    That means that the keys have an extra "module." part at the beginning.
    Use this function to remove it, allowing you to load the weights in a non-parallel network
    '''
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(string_to_remove,string_to_insert) 
        new_state_dict[new_key] = value
        
    return new_state_dict

def save_checkpoint(checkpoint_name:str,iteration_index:int,optimizer,model:nn.Module,path:str = '/work/cvcs2024/VisionWise/weights'):
    """
    Useful function to automatically save the checkpoints
    
    Args:
        checkpoint_name (str): name of the file
        iteration (int): iteration (or epoch) number
        optimizer (_type_): the optimizer
        model (_type_): the net
    """
    checkpoint_name+=('_'+str(iteration_index))
    if not checkpoint_name.endswith('.pth'):
        checkpoint_name += '.pth'
    checkpoint ={
        'model' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration_index' : iteration_index
    }
    save_path = os.path.join(path,checkpoint_name)
    torch.save(checkpoint,save_path)

def load_checkpoint(checkpoint_name:str,optimizer=None,model:nn.Module=None,iteration_index:int = None,path:str = '/work/cvcs2024/VisionWise/weights',un_parallelize = False):
    """_summary_

    Args:
        checkpoint_name (str): name of the checkpoint file
        optimizer (_type_,optional): _description_
        model (nn.Module, optional): _description_
        path (str, optional): _description_. Defaults to '/work/cvcs2024/VisionWise/weights'.

    Returns:
        int: the iteration number
    """
    list_of_all_files = os.listdir(path)
    if iteration_index != None:
        checkpoint_name = checkpoint_name + '_' + str(iteration_index)
    list_of_candidate_files = [file for file in list_of_all_files if checkpoint_name in file]
    
    if len(list_of_candidate_files) == 0:
        return 0

    max_value = float('-inf')
    load_path = None
    
    for cand in list_of_candidate_files:
        
        numbers = list(map(int, re.findall(r'\d+', cand)))
        if numbers:
            # Find the largest number in this string
            largest_number = max(numbers)
            # Update if this is the largest number found so far
            if largest_number > max_value:
                max_value = largest_number
                load_path = cand
                
    load_path = os.path.join(path,load_path)
    if not os.path.exists(load_path):
        raise RuntimeError('cannot find checkpoint file')
    checkpoint = torch.load(load_path,weights_only=False)
    
    if un_parallelize:
        model_state_dict = state_dict_adapter(checkpoint['model'],'module.','')
    else:
        model_state_dict = checkpoint['model']
    
    if model != None:
        model.load_state_dict(model_state_dict)
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['iteration_index']
    