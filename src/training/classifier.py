import numpy as np
import torch
from tqdm import tqdm
import mlflow


def train_loop(dataloader, model, loss_fn, opt, steps, epoch: int = None, device: torch.device = None):
    device = device if device is not None else torch.device("cpu")

    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    steps_max = int(np.ceil(size / dataloader.batch_size))
    steps = steps_max if steps is None else steps
    if steps > steps_max:
        steps = steps_max
    p_bar = tqdm(
        enumerate(dataloader),
        total=steps,
        desc="Training loop"
    )

    loss = 0
    for ii, (X, y) in p_bar:
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        loss.backward()
        opt.step()
        opt.zero_grad()

        if ii % 10 == 0:
            loss = loss.item()
            p_bar.set_description(f'Training loop | loss {loss:.3f}')

        if ii == steps:
            break

    mlflow.log_metric("train-loss", loss, step=epoch)


def test_loop(dataloader, model, loss_fn, steps: int = None, epoch: int = None, device: torch.device = None):
    device = device if device is not None else torch.device("cpu")

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    steps_max = int(np.ceil(size / dataloader.batch_size))
    steps = steps_max if steps is None else steps
    if steps > steps_max:
        steps = steps_max

    p_bar = tqdm(
        enumerate(dataloader),
        total=steps,
        desc="Validation loop"
    )

    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for ii, (X, y) in p_bar:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

            if ii % 10 == 0:
                p_bar.set_description(f'Validation loop | loss {test_loss / (ii+1):.3f}')

            if steps == ii:
                break

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    mlflow.log_metric("valid-loss", test_loss, step=epoch if epoch is not None else 0)
    mlflow.log_metric("valid-accuracy", correct, step=epoch if epoch is not None else 0)


if __name__ == '__main__':
    pass
