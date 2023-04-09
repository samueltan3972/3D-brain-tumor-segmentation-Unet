import torch
import numpy as np
import os
import pandas as pd

#path = 'D:/fyp/BRATS2015_Training/BRATS2015_Training/dataset_list.csv'
#path = 'D:/fyp/BRATS2015_Training/BRATS2015_Training/LGG/'
path = 'D:/fyp/BRATS2015_Testing/Testing/HGG_LGG/'
dir_list = os.listdir(path)

training_path_arr = []
ground_truth_path_arr = []

for x in dir_list:
    training_path = ''
    ground_truth_path = ''

    first_subdir = os.path.join(path, x)
    if os.path.isdir(first_subdir):

        first_subdir_list = os.listdir(first_subdir)

        for y in first_subdir_list:
            second_subdir = os.path.join(first_subdir, y)
            second_subdir_list = os.listdir(second_subdir)

            for z in second_subdir_list:
                filename = os.path.join(second_subdir, z)

                if filename.split(".")[-1] == "mha" and bool('Flair' in filename):
                    training_path = filename

                if filename.split(".")[-1] == "mha" and bool('OT' in filename):
                    ground_truth_path = filename

    training_path_arr.append(training_path)
    ground_truth_path_arr.append(ground_truth_path)

print(training_path_arr)
print(ground_truth_path_arr)
# for i in training_path_arr:
#     print(i)
#
# print("\n\n\n\n\n------------------------------------------------------\n\n\n\n\n")
#
# for j in ground_truth_path_arr:
#     print(j)

# filelist = pd.read_csv(path)
# print(filelist['filepath'][3])
#print(dir_list)