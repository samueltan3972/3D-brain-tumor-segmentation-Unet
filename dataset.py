import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from medpy.io import load


class Brats2015ImageDataset(Dataset):
    def __init__(self, path_file, prefix, transform=None, target_transform=None):

        self.img_path = pd.read_csv(path_file)
        self.img_dir_prefix = prefix
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_path['id'])

    def __getitem__(self, idx):
        train_image_data, train_image_header = load(
            os.path.join(self.img_dir_prefix, self.img_path['train filepath'][idx]))
        ground_truth_image_data, ground_truth_image_header = load(
            os.path.join(self.img_dir_prefix, self.img_path['ground truth'][idx]))
        # label = self.img_labels.iloc[idx, 1]

        # Preprocessing
        train_image_data = np.expand_dims(train_image_data, axis=0)
        train_image_data = train_image_data.astype(np.float32)
        ground_truth_image_data = np.expand_dims(ground_truth_image_data, axis=0)
        ground_truth_image_data = np.clip(ground_truth_image_data, 0, 1)
        ground_truth_image_data = ground_truth_image_data.astype(np.float32)

        if self.transform:
            train_image_data = self.transform(train_image_data)

        if self.transform:
            ground_truth_image_data = self.transform(ground_truth_image_data)

        return train_image_data, ground_truth_image_data

    def get_img_header(self, idx):
        train_image_data, train_image_header = load(
            os.path.join(self.img_dir_prefix, self.img_path['train filepath'][idx]))

        return train_image_header
