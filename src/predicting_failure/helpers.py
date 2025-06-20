import h5py
import pickle
import torch 
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

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
    '''
    Stops algorithm training early if loss metric (training/validation)
    does not improve by 'delta' over 'patience' epochs

    args:
        patience (int): Number of epochs to wait for a minimum of 'delta' improvement in loss
        delta (float): The amount by which the loss must improve each epoch to continue training
        verbose (bool): Weather the function will send debugging text to the console 
    '''
    def __init__(self, patience:int=5, delta:float=0, verbose:bool=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        '''
        Checks weather the loss of the previous run has improved by delta
        iterates a counter if not, counter will stop training when counter
        is greater than or equal to self.patience

        args:
            val_loss (float): The validation loss of the current epoch

        returns:
            None
        '''
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


# def load_data(data_path:str, n_samples:int=-1): # -> tuple[Dataoader,]:
#     '''
#     Takes in hdf5 file and returns Training and Validation data loaders

#     '''

#     if n_samples<-2:
#         raise ValueError("n_samples is <=0, Please choose a positive integer greater than 0 for sample size") 

#     try:
#         with h5py.File(data_path, 'r') as hf:
            
#             # unit_num = data_path[data_path.rfind("_")+1:data_path.rfind(".h5")]

#             if n_samples == -1:
#                 # Not selecting the Unit number and cycle number as training features, excluding the lables
#                 features = hf[f'engine_data'][:,:,2:-1]
#                 # Selecting only the RUL data to predict on
#                 labels   = hf[f'engine_data'][:,:,-1][:,-1]
#             else:
#                 # Not selecting the Unit number and cycle number as training features, excluding the lables
#                 features = hf[f'engine_data'][:n_samples,:,2:-1]
#                 # Selecting only the RUL data to predict on
#                 labels   = hf[f'engine_data'][:n_samples,:,-1][:,-1]




#         print(f"load_data(), Features shape:{features.shape}, Labels shape: {labels.shape}")
#         print(features[0])
#         print(labels[0])

#         # Create Dataset and DataLoader
#         dataset = MyDataset(features, labels)

#         # Define split ratio
#         dataset_size = len(dataset)
#         train_size = int(0.8 * dataset_size)
#         val_size = dataset_size - train_size

#         # Split the dataset
#         train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



#         # Create DataLoaders
#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
#         val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


#         return train_loader, val_loader
    
#     except ValueError as e:
#         print(f"Error: {e}")

def persistify_scaling_object(scaling_object, scaling_object_name: str) -> None:
    '''
    Will save an instance of the scaling function used to preprocess the data
    '''
    
    filename = "./scalars/"+scaling_object_name

    with open(filename, 'wb') as file:
        pickle.dump(scaling_object, file)

    print(f"Saved scalar at : {filename} ")


def load_scaling_object(scalar_path: str) -> StandardScaler:
    '''
    Load up the existing scalar used to train the model
    allows for live preprocessing of incoming data
    '''

    with open(scalar_path, 'rb') as file:
        loaded_scaler = pickle.load(file)

    

    return loaded_scaler


def load_data(data_path: str, n_samples: int = -1, batch_size:int = 64):
    '''
    Loads data from HDF5, scales features using training set stats,
    splits into train and validation, returns DataLoaders.
    '''
    if n_samples < -2:
        raise ValueError("n_samples must be positive or -1 (for full dataset).")

    try:
        with h5py.File(data_path, 'r') as hf:
            if n_samples == -1:
                features = hf['engine_data'][:, :, 2:-1]  # shape: (N, T, F)
                labels   = hf['engine_data'][:, :, -1][:, -1]  # shape: (N,)
            else:
                features = hf['engine_data'][:n_samples, :, 2:-1]
                labels   = hf['engine_data'][:n_samples, :, -1][:, -1]

        print(f"load_data(), Features shape: {features.shape}, Labels shape: {labels.shape}")

        # Split indices first
        dataset_size = features.shape[0]
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        train_features = features[:train_size]  # (N_train, T, F)
        train_labels   = labels[:train_size]
        val_features   = features[train_size:]
        val_labels     = labels[train_size:]

        # Reshape for scaler: (N*T, F)
        N_train, T, F = train_features.shape
        train_features_flat = train_features.reshape(-1, F)

        scaler = StandardScaler()
        scaler.fit(train_features_flat)  # Fit on training data only
        persistify_scaling_object(scaler, "scalar_FD001.pkl")

        # Apply transform to both train and val
        train_features_scaled = scaler.transform(train_features_flat).reshape(N_train, T, F)
        val_features_scaled = scaler.transform(val_features.reshape(-1, F)).reshape(val_features.shape)

        # Convert to tensors
        train_tensor_x = torch.tensor(train_features_scaled, dtype=torch.float32)
        train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)
        val_tensor_x = torch.tensor(val_features_scaled, dtype=torch.float32)
        val_tensor_y = torch.tensor(val_labels, dtype=torch.float32)

        # Create Datasets
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        val_dataset   = TensorDataset(val_tensor_x, val_tensor_y)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader #, scaler  # Return scaler too if needed later
    
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
            labels   = hf[f'engine_data'][:,:,-1][:,-1]
        else:
            # Not selecting the Unit number and cycle number as training features, excluding the lables
            features = hf[f'engine_data'][:n_samples,:,2:-1]
            # Selecting only the RUL data to predict on
            labels   = hf[f'engine_data'][:n_samples,:,-1,][:,-1]

    print(f"load_eval_data(), Features shape:{features.shape}, Labels shape: {labels.shape}")

    # Create Dataset and DataLoader
    dataset = MyDataset(features, labels)


    eval_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)



    return eval_loader

def load_eval_data_singleRUL(data_path:str, scalar_path:str, batch_size:int, n_samples:int=-1):
    '''
    Takes in hdf5 file and returns Training and Validation data loaders

    '''

    with h5py.File(data_path, 'r') as hf:
        if n_samples == -1:
            # Not selecting the Unit number and cycle number as training features, excluding the lables
            features = hf[f'engine_data'][:,:,2:-1]
            # Selecting only the RUL data to predict on
            labels   = hf[f'engine_data'][:,:,-1][:,-1]
        else:
            # Not selecting the Unit number and cycle number as training features, excluding the lables
            features = hf[f'engine_data'][:n_samples,:,2:-1]
            # Selecting only the RUL data to predict on
            labels   = hf[f'engine_data'][:n_samples,:,-1,][:,-1]

    print(f"load_eval_data(), Features shape:{features.shape}, Labels shape: {labels.shape}")

        # Reshape for scaler: (N*T, F)
    N_eval, T, F = features.shape
    eval_features_flat = features.reshape(-1, F)
    
    # Need to scale data appropriately
    scalar = load_scaling_object(scalar_path=scalar_path)
    # Apply transform to both eval and val
    eval_features_scaled = scalar.transform(eval_features_flat).reshape(N_eval, T, F)

    # Convert to tensors
    eval_tensor_x = torch.tensor(eval_features_scaled, dtype=torch.float32)
    eval_tensor_y = torch.tensor(labels, dtype=torch.float32)

    eval_dataset = TensorDataset(eval_tensor_x, eval_tensor_y)

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)



    return eval_loader

