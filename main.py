from multiprocessing import freeze_support

import torch
import numpy as np
import sys
import os
import pandas as pd
from torchvision import transforms
from train import train

import dataset
from medpy.io import load
from torch.utils.data import DataLoader, random_split
from unet_model import UNet
from evaluate import evaluate

if __name__ == '__main__':
    freeze_support()

# Load dataset
#path = 'D:/fyp/BRATS2015_Training/BRATS2015_Training/dataset_list.csv'
path = 'D:/fyp/fyp2-selected dataset train/dataset_list.csv'
brats2015_dataset = dataset.Brats2015ImageDataset(path, 'D:/fyp/fyp2-selected dataset train/')

# Split training and testing set
# test_percent = 0.2
# n_test = int(len(brats2015_dataset) * test_percent)
# n_train = len(brats2015_dataset) - n_test
# brats2015_trainset, brats2015_testset = random_split(brats2015_dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))
# np.set_printoptions(threshold=sys.maxsize)

brats2015_trainset = brats2015_dataset
# Print dataset info
print('Number of image in Brats 2015 training dataset:', len(brats2015_trainset))
# print('Number of image in Brats 2015 testing dataset:', len(brats2015_testset))
print(len(brats2015_trainset)) # The image comes with shape of (c, x, y, z) as grayscale image

# Create dataloader
brats2015_trainloader = DataLoader(brats2015_trainset, batch_size=1, shuffle=True) # , num_workers=2)
# brats2015_testloader = DataLoader(brats2015_testset, batch_size=1, shuffle=True) # , num_workers=2)

# train_features, train_labels = next(iter(brats2015_trainloader))
#
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# Load the network
unet_model = UNet(1, 1, use_BPB=True)
#print(unet)

# Train the model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#unet_model.to(device)
device = 'cpu'
print(device)
# torch.cuda.empty_cache()


train(unet_model, brats2015_trainloader, device, num_epoch=5,  momentum=0.99,
                                                  dataset_desc='unet_bpb', savedModel=True)

# Evaluate on testing set
# evaluate(unet_model, brats2015_testloader, device=device)


# filelist = pd.read_csv(path)
# train_image_data, train_image_header = load(filelist['train filepath'][0])
#
# test, test2 = load('D:/fyp/brain tumor/brain_tumor_dataset/no/5 no.jpg')
#
# print(test.shape)
# print(train_image_data.shape)

