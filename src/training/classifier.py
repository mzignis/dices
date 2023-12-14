import numpy as np
import torch
from tqdm import tqdm
import mlflow

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(dataloader, model, loss_fn, opt):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    p_bar = tqdm(
        enumerate(dataloader),
        total=int(np.ceil(size / dataloader.batch_size)),
        desc="Training loop"
    )

    loss = 0
    for batch, (X, y) in p_bar:
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        opt.step()
        opt.zero_grad()

        if batch % 10 == 0:
            loss = loss.item()
            p_bar.set_description(f'Training loop | loss {loss:.3f}')

    mlflow.log_metric("train-loss", loss)


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    mlflow.log_metric("valid-loss", test_loss)
    mlflow.log_metric("valid-accuracy", correct)


if __name__ == '__main__':
    pass
