import h5py
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_tensor, y_tensor



def load_data(data_path:str):
    with h5py.File(data_path, 'r') as hf:
        unit_num = data_path[data_path.rfind("_")+1:data_path.rfind(".h5")]
        features = hf[f'unit_{unit_num}'][:,2:-1,:]
        labels = hf[f'unit_{unit_num}'][:,-1,:]

    # Create Dataset and DataLoader
    dataset = MyDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    return train_loader


# def format_data(data_path:str):
#     '''
#     Will load from pandas to hdf5 in LSTM format
#     '''