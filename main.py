import argparse
import datetime
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.data import DiceImageDataset
from src.training.classifier import train_loop, test_loop
from src.models.simple_cnn import SimpleCNN


# -------- parse arguments --------
parser = argparse.ArgumentParser(description='Train a classifier')
parser.add_argument('--model', type=str, help='Path to model file', required=True)
parser.add_argument('--data', type=str, help='Path to data directory', required=True)

parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=1e-3)
parser.add_argument('--training_steps', type=int, help='Number of traininig steps per epoch', default=None)
parser.add_argument('--validation_steps', type=int, help='Number of validation steps per epoch', default=None)
parser.add_argument('--device', type=str, help='Device to use')

parser.add_argument('--output', type=str, help='Path to output directory', default=None)

parser.add_argument('--name', type=str, help='Name of the run', default=None)

args = parser.parse_args()

# -------- device --------
if args.device:
    device = torch.device(args.device)
else:
    is_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if is_available else "cpu")

print(f"Using {device} device")


# -------- mlflow --------
experiment_name = "dice_classifier"
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id
run_name = args.name if args.name else f"{datetime.datetime.now().strftime('%Y%d%m%H%M%S')}-dice-classifier"
print("experiment_id:", experiment_id)
print("run_name:", run_name)
mlflow.autolog()


# -------- hyperparameters --------
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs

# -------- load model --------
model_filepath = Path(args.model).resolve()
print("model exists:", model_filepath.exists())
model_dice = torch.load(model_filepath).to(device)

# -------- dataset --------
data_dirpath = Path(args.data).resolve()
ds_train = DiceImageDataset(data_dirpath / 'train')
ds_valid = DiceImageDataset(data_dirpath / 'valid')

dataloader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

# -------- initialize the loss function --------
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_dice.parameters(), lr=learning_rate)

# -------- init mlflow run --------
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model_filepath", model_filepath)
    mlflow.log_param("data_dirpath", data_dirpath)

    # -------- training loop --------
    output_filepath = Path(args.output).resolve() if args.output else Path("models") / f"{run_name}.pb"
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(
            dataloader_train, model_dice, loss_function, optimizer,
            steps=args.training_steps, epoch=epoch, device=device,
        )
        test_loop(
            dataloader_valid, model_dice, loss_function,
            steps=args.validation_steps, epoch=epoch, device=device,
        )
        if output_filepath:
            torch.save(model_dice, output_filepath)
        print()

    # -------- save model --------
    if output_filepath:
        torch.save(model_dice, output_filepath)

# -------- finish --------
print("Done!")
