import os

def num_files(data_class, set_name):
    '''
    Finds the number of files for some data class and set name.
    '''
    dir = './data/working/'+str(data_class)+'/'+set_name
    return len(os.listdir(dir))

def num_coords(data_class):
    return data_class**2