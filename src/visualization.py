from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import DiceImageDataset

model_filepath = Path('../models/first_model.pb')
model = torch.load(model_filepath)


ds_valid = DiceImageDataset('../data/dice/valid')
dataloader_valid = DataLoader(ds_valid, batch_size=16, shuffle=True)

images, targets = next(iter(dataloader_valid))
targets_pred = np.argmax(model(images).detach().numpy(), axis=1)

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))

for ii in range(4):
    for jj in range(4):
        idx = ii * 4 + jj
        img = images[idx].permute(2, 1, 0)
        img = img * 255
        target = targets[idx]
        ax[ii][jj].imshow(img.numpy().astype(np.uint8))
        ax[ii][jj].set_title(f'target {target} | pred {targets_pred[idx]}')
        ax[ii][jj].set_xticks([])
        ax[ii][jj].set_yticks([])

plt.show()
