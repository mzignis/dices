import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.data import DiceImageDataset
from src.models import *


is_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
device = torch.device("mps" if is_available else "cpu")
print(f"Using {device} device")


class Visualizator(object):
    def __init__(self, model_filepath: Path, data_dirpath: Path, label_mapper: dict = None, shuffle: bool = True):
        self.model = torch.load(model_filepath).to(device)
        self.model.eval()

        self.ds = DiceImageDataset(data_dirpath)
        self.dataloader = DataLoader(self.ds, batch_size=16, shuffle=shuffle)

        self.label_mapper = label_mapper if label_mapper is not None else {
            0: 'd10', 1: 'd12', 2: 'd20', 3: 'd4', 4: 'd6', 5: 'd8'
        }

    def visualize(self, rows: int = 4, cols: int = 4):
        images, targets = next(iter(self.dataloader))
        targets_pred = self.model(images.to(device)).argmax(1).cpu().numpy()

        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
        for ii in range(rows):
            for jj in range(cols):
                ax[ii][jj].set_xticks([])
                ax[ii][jj].set_yticks([])
                idx = ii * 4 + jj
                if idx >= targets_pred.shape[0]:
                    break
                img = images[idx].permute(2, 1, 0)
                img = img * 255
                target = targets[idx]
                ax[ii][jj].imshow(img.numpy().astype(np.uint8))
                ax[ii][jj].set_title(
                    f'{self.label_mapper[int(target)]} | pred {self.label_mapper[int(targets_pred[idx])]}'
                )

        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model file', required=True)
    parser.add_argument('--data', type=str, help='Path to data directory', required=True)
    args = parser.parse_args()

    model_filepath = Path(args.model).resolve()
    data_dirpath = Path(args.data).resolve()

    visualizator = Visualizator(model_filepath, data_dirpath)
    visualizator.visualize()
