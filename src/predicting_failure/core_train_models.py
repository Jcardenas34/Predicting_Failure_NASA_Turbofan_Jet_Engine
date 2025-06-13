from predicting_failure.models import Recurrent


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
    model.train()
    

    # Train using num_epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}')

    return model
