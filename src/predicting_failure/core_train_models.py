import time
import h5py
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from predicting_failure.models import Recurrent
from predicting_failure.helpers import EarlyStopping
from predicting_failure.helpers import load_data
from predicting_failure.models import Recurrent, SingleRUL

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
def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs:int = 10, patience:int = 5, delta:float=0.001):
    """
    Train the model using the provided data loader, criterion, and optimizer.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for updating model parameters.
        num_epochs: Number of epochs to train the model.
    
    returns:
        model: A trained pytorch model defined in models.py

    """
    # Set model to training mode

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
        model.to(device)
    elif torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
        model.to(device)
    else:
        print("No GPU, using CPU")
        device = torch.device("cpu")
        model.to(device)

    
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

    # Train using num_epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        val_loss = 0.0

        # Training phase
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape, targets.shape)
            print(outputs[0].item(), targets[0].item())
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = loss_function(output, target)
                val_loss += loss.item()
        
        # Average validation loss
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}, Average Val Loss: {val_loss / len(val_loader):.4f}')
        model_path = f"models/RUL_regressor_unit1_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)

        # Check early stopping condition
        early_stopping.check_early_stop(val_loss)
        
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

    return model


def evaluate_model(model_path:str, data_path:str, eval_loader, loss_function):
    '''
    Evaluates model input where input is passed as an hdf5 file

    args:
        model_path (str):
        data_path (str):
        eval_loader:
        loss_function ()::

    '''

    # 1. Initialize the model
    # model = Recurrent()
    model = SingleRUL()




    if torch.cuda.is_available():
        print("Using GPU")
        state_dict = torch.load(model_path)
    else:
        print("No GPU, using CPU")
        state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Initialize variables to store evaluation metrics
    total_loss = 0
    all_predictions = []
    all_labels = []

    model.eval()
    # Perform evaluation
    sample = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            # predictions = torch.argmax(outputs, dim=1)

            # all_predictions.extend(predictions.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            print("Predictions, Labels")
            for i,j in zip(outputs[sample],labels[sample]):
                print(f"{i.item():2f}, {j.item():2f}")
            sample+=1


    # Calculate average loss and accuracy
    # average_loss = total_loss / len(eval_loader)


    # # accuracy = accuracy_score(all_labels, all_predictions)

    # # Print the results
    # print(f"Average Test Loss: {average_loss:.4f}")
    # print(f"Test Accuracy: {accuracy:.4f}")


