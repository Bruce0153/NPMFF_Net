import os
import pickle
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class ModelNet40FewShot(Dataset):
    def __init__(
        self,
        root,
        way: int,
        shot: int,
        fold: int,
        split: str = "train",
        normals: bool = True,
        transforms: Optional = None,
    ):
        self.normals = normals
        self.transforms = transforms

        assert split == "train" or split == "test"

        with open(os.path.join(root, f"{way}way_{shot}shot", f"{fold}.pkl"), "rb") as f:
            self.data = pickle.load(f)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        points, label, _ = self.data[index]
        if self.transforms is not None:
            points = self.transforms(points)

        if not self.normals:
            points = points[:, :3]

        return points, label


class ModelNet40FewShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        way: int,
        shot: int,
        fold: int,
        data_dir: str = "../data/ModelNetFewshot",
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ModelNet40FewShot(self.hparams.data_dir, way=self.hparams.way, shot=self.hparams.shot, fold=self.hparams.fold, split="train")  # type: ignore
        self.test_dataset = ModelNet40FewShot(self.hparams.data_dir, way=self.hparams.way, shot=self.hparams.shot, fold=self.hparams.fold, split="test")  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=True,  # type: ignore
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
        )

    @property
    def num_classes(self):
        return self.hparams.way  # type: ignore

if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNet40FewShot('/home/dell/ZHL/PointSS/data/ModelNetFewshot', split="train", way=10, shot=10, fold=2, transforms=transforms)
    print(dset[90][0])
    print(dset[90][1])
    print(len(dset))
