from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize


class LabelTransformer:

    def __init__(self, labels: list):
        self.unique = sorted(np.unique(labels))
        self.values_dict = {lbl_name: lbl_idx for lbl_idx, lbl_name in enumerate(self.unique)}
        self.key_dict = {lbl_idx: lbl_name for lbl_idx, lbl_name in enumerate(self.unique)}

    def __call__(self, label_name):
        return self.label_to_num(label_name)

    def label_to_num(self, label_name):
        return self.values_dict[label_name]

    def num_to_label(self, num):
        return self.key_dict[int(num)]


class DiceImageDataset(Dataset):
    def __init__(self, img_dirpath: [Path, str], img_extension: str = 'jpg', transform=None):
        # -------- images --------
        self.img_dirpath = Path(img_dirpath).resolve()
        self.img_extension: str = img_extension
        self.images = [(x, x.parent.name) for x in self.img_dirpath.glob(f'*/**/*.{self.img_extension}')]

        # -------- transformers ---------
        self.transform = Compose([Resize(224), Normalize(127.5, 127.5)])
        self.target_transform = LabelTransformer([x[1] for x in self.images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx][0]
        label = self.images[idx][1]

        image = read_image(str(img_path)).float()

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    ds_train = DiceImageDataset('../data/dice/train')
    dataloader_train = DataLoader(ds_train, batch_size=4, shuffle=True)

    train_images, train_labels = next(iter(dataloader_train))
    print(train_labels)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(5, 2))
    for ii, (img, lbl) in enumerate(zip(train_images, train_labels)):
        ax[ii].imshow(img.permute(1, 2, 0))
        ax[ii].set_title(ds_train.target_transform.num_to_label(lbl))
        ax[ii].set_xticks([])
        ax[ii].set_yticks([])

    plt.show()
