import h5py
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split

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
    '''
    Takes in hdf5 file and returns Training and Valudation data loaders

    '''
    with h5py.File(data_path, 'r') as hf:
        unit_num = data_path[data_path.rfind("_")+1:data_path.rfind(".h5")]
        # Not selecting the Unit number and cycle number as trainigng features, excluding the lables
        features = hf[f'unit_{unit_num}'][:,2:-1,:]
        # Selecting only the RUL data to predict on
        labels = hf[f'unit_{unit_num}'][:,-1,:]



    # Create Dataset and DataLoader
    dataset = MyDataset(features, labels)

    # Define split ratio
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)



    return train_loader, val_loader


# def format_data(data_path:str):
#     '''
#     Will load from pandas to hdf5 in LSTM format
#     '''