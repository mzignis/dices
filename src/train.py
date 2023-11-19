import numpy as np

from src.models import SimpleCNN
from src.data import DiceImageDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


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
    for batch, (X, y) in p_bar:
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        opt.step()
        opt.zero_grad()

        if batch % 10 == 0:
            loss = loss.item()
            p_bar.set_description(f'Training loop | loss {loss:.3f}')


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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    # -------- load model --------
    model_dice = SimpleCNN(6)

    # -------- dataset --------
    ds_train = DiceImageDataset('../data/dice/train')
    ds_valid = DiceImageDataset('../data/dice/valid')

    dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

    # Initialize the loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_dice.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader_train, model_dice, loss_function, optimizer)
        test_loop(dataloader_valid, model_dice, loss_function)
        print()
    print("Done!")

    torch.save(model_dice.state_dict(), Path('../models/first_model.pth'))
