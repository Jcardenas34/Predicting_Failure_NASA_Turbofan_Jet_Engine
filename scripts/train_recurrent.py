import torch.nn as nn
from torch.optim import Adam
from predicting_failure.models import Recurrent
from predicting_failure.helpers import load_data
from predicting_failure.core_train_models import train_model


def main():
    """
    Performs the actual training of the LSTM based model
    """

    # Retrieve the model
    model = Recurrent(n_features=24) 
    loss_function = nn.L1Loss() # Appropriate becase we are predicting RUL, want to predict integer
    optimizer = Adam(model.parameters(), lr=0.001)

    data_path =  "/Users/chiral/git_projects/Predicting_Failure_NASA_Turbofan_Jet_Engine/data/unit_1.h5"
    loader = load_data(data_path)

    train_model(
        model=model,
        train_loader=loader,  # Replace with actual DataLoader
        loss_function=loss_function,     # Replace with actual loss function
        optimizer=optimizer,     # Replace with actual optimizer
        num_epochs=2       # Set the number of epochs
    )


if __name__ == "__main__":
    main()
