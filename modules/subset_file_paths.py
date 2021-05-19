## To get paths, just import this file and do subset_file_paths.paths to get a list of absolute paths
import os

if "\\" in os.getcwd():
    path = '\\'.join( os.getcwd().split('\\')[:-1] ) + f'\\Data\\subsets'
else:
    path = '/'.join(os.getcwd().split('/')[:-1] ) + '/Data/subsets'

subsets = next(os.walk(path))[2]
paths = [path.replace('\\', '/') + '/' + file for file in subsets if file[0] == 'n']
