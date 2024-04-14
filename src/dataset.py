import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from src.generate_folds import make_folds
from src.augmentations import train_augmentations, val_augmentations, test_augmentations


TARGETS = ['Археология', 'Оружие', 'Прочие', 'Нумизматика', 'Фото, негативы',
           'Редкие книги', 'Документы', 'Печатная продукция', 'ДПИ',
           'Скульптура', 'Графика', 'Техника', 'Живопись',
           'Естественнонауч.коллекция', 'Минералогия']


class ClassifierDataset(Dataset):
    def __init__(self, data_path, mode, fold: int):
        assert mode in ["train", "eval", "test"]
        self.data_path = data_path
        data = pd.read_csv(data_path, sep=";")

        if "fold" not in data.columns and mode != "test":
            data = make_folds(data, random_state=21, save=data_path.split(".")[0] + "_unique.csv")

        if mode == "train":
            self.data = data[data.fold != fold]
        elif mode == 'eval':
            self.data = data[(data.fold == fold)]
        else:
            self.data = data

        self.mode = mode

        if self.mode == "train":
            self.augs = train_augmentations()
        elif self.mode == "eval":
            self.augs = val_augmentations()
        elif self.mode == "test":
            self.augs = test_augmentations()

        if self.mode == "train":
            assert len(self.data[self.data.fold == fold]) == 0
        elif mode == 'eval':
            assert len(self.data[self.data.fold != fold]) == 0

    def __len__(self):
        return self.data.shape[0]

    def transform(self, image, mode):
        image = self.augs(image=image)["image"]
        return image
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.data_path.split("/")[0], str(row["object_id"]), row["img_name"])
        image = cv2.imread(image_path)
        if image is None:
            image = np.asarray(Image.open(image_path))
            if len(image.shape) == 2:
                image = np.stack([image, image, image]).transpose((1,2,0))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "test":
            target = -1
        else:
            target = TARGETS.index(row["group"])

        image = self.transform(image, self.mode)
        
        sample = {}
        sample["image"] = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (3, 512, 512) shape
        sample["label"] = target
        sample["object_id"] = row["object_id"]
        sample["img_name"] = row["img_name"]

        return sample
        