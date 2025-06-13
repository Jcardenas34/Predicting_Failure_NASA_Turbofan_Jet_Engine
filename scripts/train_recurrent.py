import torch.nn as nn
import torch
from torch.optim import Adam
from predicting_failure.models import Recurrent
from predicting_failure.helpers import load_data
from predicting_failure.core_train_models import train_model

import logging
import functools
import argparse


def main(args):
    """
    Performs the actual training of the LSTM based model
    """

    # Retrieve the model
    model = Recurrent(n_features=24) 
    loss_function = nn.L1Loss() # Appropriate becase we are predicting RUL, want to predict integer
    optimizer = Adam(model.parameters(), lr=0.001)

    # data_path =  "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/unit_1.h5"
    train_loader, val_loader = load_data(args.data_path)

    model = train_model(
        model=model,
        train_loader = train_loader,  # Replace with actual DataLoader
        val_loader = val_loader,
        loss_function=loss_function,     # Replace with actual loss function
        optimizer=optimizer,     # Replace with actual optimizer
        num_epochs=args.epochs       # Set the number of epochs
    )

    # torch.save(model, 'RUL_regressor_unit_1.pth')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, required=True)
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10)
    args = parser.parse_args()

    main(args)
