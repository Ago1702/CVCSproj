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