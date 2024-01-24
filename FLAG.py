import torch

path_to_data = '../p36/data'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('[INFO] You are using {}\n'.format(device))
inf = 99999999.
validation_data_flush_interval = 5
