import h5py
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
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

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


def load_data(data_path:str, n_samples:int=-1): # -> tuple[Dataoader,]:
    '''
    Takes in hdf5 file and returns Training and Validation data loaders

    '''

    if n_samples<-2:
        raise ValueError("n_samples is <=0, Please choose a positive integer greater than 0 for sample size") 

    try:
        with h5py.File(data_path, 'r') as hf:
            
            # unit_num = data_path[data_path.rfind("_")+1:data_path.rfind(".h5")]

            if n_samples == -1:
                # Not selecting the Unit number and cycle number as training features, excluding the lables
                features = hf[f'engine_data'][:,:,2:-1]
                # Selecting only the RUL data to predict on
                labels   = hf[f'engine_data'][:,:,-1]
            else:
                # Not selecting the Unit number and cycle number as training features, excluding the lables
                features = hf[f'engine_data'][:n_samples,:,2:-1]
                # Selecting only the RUL data to predict on
                labels   = hf[f'engine_data'][:n_samples,:,-1]



        print(f"load_data(), Features shape:{features.shape}, Labels shape: {labels.shape}")
        print(features[0])
        print(labels[0])

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
    
    except ValueError as e:
        print(f"Error: {e}")


def collate_fn(batch):
    # batch is a list of (x_tensor, y_tensor) tuples
    xs, ys = zip(*batch)

    # Find true lengths based on non-zero rows
    lengths = torch.tensor([x.shape[0] if x.ndim == 2 else sum((x.abs().sum(dim=1) != 0).int()).item() for x in xs])
    # print(lengths)
    # Pad xs to longest in batch
    xs_padded = pad_sequence(xs, batch_first=True)  # [B, T_max, F]
    ys_padded = pad_sequence(ys, batch_first=True)  # [B, T_max] or [B, T_max, 1]

    return xs_padded, ys_padded, lengths

def load_eval_data(data_path:str, n_samples:int=-1):
    '''
    Takes in hdf5 file and returns Training and Validation data loaders

    '''
    with h5py.File(data_path, 'r') as hf:
        if n_samples == -1:
            # Not selecting the Unit number and cycle number as training features, excluding the lables
            features = hf[f'engine_data'][:,:,2:-1]
            # Selecting only the RUL data to predict on
            labels   = hf[f'engine_data'][:,:,-1]
        else:
            # Not selecting the Unit number and cycle number as training features, excluding the lables
            features = hf[f'engine_data'][:n_samples,:,2:-1]
            # Selecting only the RUL data to predict on
            labels   = hf[f'engine_data'][:n_samples,:,-1]

    print(f"load_eval_data(), Features shape:{features.shape}, Labels shape: {labels.shape}")

    # Create Dataset and DataLoader
    dataset = MyDataset(features, labels)


    eval_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)



    return eval_loader

