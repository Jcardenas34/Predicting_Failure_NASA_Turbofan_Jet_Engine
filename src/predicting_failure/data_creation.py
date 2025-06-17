

import pandas as pd
import numpy as np
import h5py





def create_train_data(data_path:str, out_name:str="train_engine_data.h5", window_size:int=150, n_engines:int = 1):
    '''
    Creates an hdf5 data from the data_path.csv file

    args:
        data_path (str): File path to data
        out_name (str): What the training data will be named "train_engine_data.h5" is the default
        window_size (int): How large the RUL prediction data is
        n_engines (int): How many engines you want to use in the dataset creation (There are 100 max)
    
    returns:
        df (hdf5 file): An hdf5 file type with the "engine_data" key
        
    '''


    # Conducting exploratory analysis on engine data
    df = pd.read_csv(data_path, delimiter=' ')

    column_names = ["unit_number","cycle_number","operational_setting_1","operational_setting_2","operational_setting_3"]

    n_cols = len(df.columns)
    print(f"Number of Columns: {n_cols}")


    # The rest of the columns are sensor measurements, let's label the remaining by an index 
    for k in range(1,len(df.columns)-len(column_names)+1):
        column_names.append("sensor_measurement_{}".format(k))

    df.columns = column_names
    # n_engines = len(df["unit_number"].unique())
    print(f"Number of Units: {n_engines}")


    # Getting the max cyle number of each unit
    grouped_data = df.groupby("unit_number")["cycle_number"].max().reset_index()

    # renaming column
    grouped_data = grouped_data.rename(columns={"cycle_number":"max_cycles"})

    # Creating a dictionary of the max number of cycles run
    max_dict = grouped_data.set_index('unit_number')['max_cycles'].to_dict()

    # Adding True RUL to dataframe
    df["RUL"] = df.apply(lambda row: max_dict[row["unit_number"]]-row["cycle_number"], axis=1)

    # Lets drop the irrelevant columns
    df.drop(["sensor_measurement_22","sensor_measurement_23"], axis=1, inplace=True)

    windows =[]
    for i in range(1, n_engines+1):  # or loop over all unit_numbers
        print(f"Unit: {i}")
        
        sub_sample = df[df["unit_number"] == i].copy()
        print("Original shape:", sub_sample.shape)

        # Determine how many padding rows are needed
        num_to_pad = window_size - len(sub_sample)

        if num_to_pad > 0:
            print(f"Padding by: {num_to_pad}")
            # Create a DataFrame with zero rows that match sub_sample columns
            padding_df = pd.DataFrame(
                data=np.zeros((num_to_pad, sub_sample.shape[1])),
                columns=sub_sample.columns
            )

            # Optionally fill in identifier columns, if needed:
            padding_df["unit_number"] = i
            padding_df["cycle_number"] = np.arange(-num_to_pad, 0)  # Dummy values

            # Append padded rows
            sub_sample = pd.concat([padding_df, sub_sample], ignore_index=True)
            print("Final shape:", sub_sample.shape)
            windows.append(sub_sample.to_numpy())
        
        else:
            for i in range(len(sub_sample) - window_size + 1):
                # print(sub_sample.iloc[i:i+window_size].to_numpy().shape)
                windows.append(sub_sample.iloc[i:i+window_size].to_numpy())


    # windows = [df.iloc[i:i+window_size].to_numpy() for i in range(len(df) - window_size + 1)]

    with h5py.File(f'./data/{out_name}', 'w') as hf:
        hf.create_dataset('engine_data', data=windows)

    with h5py.File(f'./data/{out_name}', 'r') as hf:
        loaded_array = hf['engine_data'][:]
    
    print(f"Final Shape: {loaded_array.shape}")



    return hf



def create_test_data(data_path:str, true_rul_path:str, out_name:str="test_engine_data.h5", window_size:int=150, n_engines:int = 1):
 # Conducting exploratory analysis on engine data
    df = pd.read_csv(data_path, delimiter=' ')

    column_names = ["unit_number","cycle_number","operational_setting_1","operational_setting_2","operational_setting_3"]

    n_cols = len(df.columns)
    print(f"Number of Columns: {n_cols}")


    # The rest of the columns are sensor measurements, let's label the remaining by an index 
    for k in range(1,len(df.columns)-len(column_names)+1):
        column_names.append("sensor_measurement_{}".format(k))

    df.columns = column_names
    # n_engines = len(df["unit_number"].unique())
    print(f"Number of Units: {n_engines}")

    df.drop(["sensor_measurement_22","sensor_measurement_23"], axis=1, inplace=True)
    
    # Getting the max cyle number of each unit
    grouped_data = df.groupby("unit_number")["cycle_number"].max().reset_index()

    # renaming column
    grouped_data = grouped_data.rename(columns={"cycle_number":"max_cycles"})

    # Creating a dictionary of the max number of cycles run
    max_dict = grouped_data.set_index('unit_number')['max_cycles'].to_dict()


    # Loading the true RUL after last observed cycle
    true_RUL_df = pd.read_csv(true_rul_path, header=None, names=["RUL"])
    true_RUL_df = true_RUL_df.reset_index()
    true_RUL_df["unit_number"] = true_RUL_df["index"].apply(lambda row: row+1)

    rul_dict = true_RUL_df.set_index('unit_number')['RUL'].to_dict()



    # We want the true RUL to be 
    # True_RUL + Max_cycles - current_cycle, so that if current_cycle== Max_cycle we get the true RUL for the last run
    df["true_RUL"] = df[["unit_number","cycle_number"]].apply(lambda row: rul_dict[row["unit_number"]] + max_dict[row["unit_number"]] - row["cycle_number"], axis=1)

    windows = []
    for i in range(1, n_engines):  # or loop over all unit_numbers
        print(f"Unit: {i}")
        
        sub_sample = df[df["unit_number"] == i].copy()
        print("Original shape:", sub_sample.shape)

        # Determine how many padding rows are needed
        num_to_pad = window_size - len(sub_sample)

        if num_to_pad > 0:
            # Create a DataFrame with zero rows that match sub_sample columns
            padding_df = pd.DataFrame(
                data=np.zeros((num_to_pad, sub_sample.shape[1])),
                columns=sub_sample.columns
            )

            # Optionally fill in identifier columns, if needed:
            padding_df["unit_number"] = i
            padding_df["cycle_number"] = np.arange(-num_to_pad, 0)  # Dummy values

            # Append padded rows
            sub_sample = pd.concat([padding_df, sub_sample], ignore_index=True)
            print("Final shape:", sub_sample.shape)
            windows.append(sub_sample.to_numpy())
        
        else:
            for i in range(len(sub_sample) - window_size + 1):
                # print(sub_sample.iloc[i:i+window_size].to_numpy().shape)
                windows.append(sub_sample.iloc[i:i+window_size].to_numpy())


    with h5py.File(f'./data/{out_name}', 'w') as hf:
        hf.create_dataset('engine_data', data=windows)

    with h5py.File(f'./data/{out_name}', 'r') as hf:
        loaded_array = hf['engine_data'][:]
    
    print(f"Final Shape: {loaded_array.shape}")


def create_hdf5(unit_num, np_array):
    '''
    Creating the hdf5 dataset according to the unit number
    '''
    # Store the array into an HDF5 file
    with h5py.File(f'../data/train_unit_{unit_num}.h5', 'w') as hf:
        hf.create_dataset(f'engine_data', data=np_array)
    
    # Load the array from the HDF5 file
    with h5py.File(f'../data/train_unit_{unit_num}.h5', 'r') as hf:
        loaded_array = hf[f'engine_data'][:]
    
    # Verify the loaded array
    print(np.array_equal(np_array, loaded_array)) # Should print True

