import torch.nn as nn
import torch
from torch.optim import Adam
from predicting_failure.models import Recurrent
from predicting_failure.helpers import load_data
from predicting_failure.core_train_models import train_model

import logging
import functools

def log_output(log_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__name__)
            logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler(log_file, mode='a')
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            try:
                result = func(*args, **kwargs)
                logger.info(f"Function {func.__name__} called with args: {args}, kwargs: {kwargs} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise
            finally:
                logger.removeHandler(file_handler)
                logger.removeHandler(console_handler)
        return wrapper
    return decorator


@log_output("RUL_log.log")
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

    model = train_model(
        model=model,
        train_loader=loader,  # Replace with actual DataLoader
        loss_function=loss_function,     # Replace with actual loss function
        optimizer=optimizer,     # Replace with actual optimizer
        num_epochs=2       # Set the number of epochs
    )

    # torch.save(model, 'RUL_regressor_unit_1.pth')

if __name__ == "__main__":
    main()
