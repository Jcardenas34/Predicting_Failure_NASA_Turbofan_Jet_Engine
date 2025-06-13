import torch
from predicting_failure.models import Recurrent

import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper


@time_function
def train_model(model, train_loader, loss_function, optimizer, num_epochs=10):
    """
    Train the model using the provided data loader, criterion, and optimizer.

    :param model: The model to train.
    :param train_loader: DataLoader for training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer for updating model parameters.
    :param num_epochs: Number of epochs to train the model.
    """
    # Set model to training mode

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("No GPU, using CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    

    # Train using num_epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs.to(device)
            targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}')
        model_path = f"models/RUL_regressor_unit1_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)

    return model
