import json
import math
import os

import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from loss import BinaryDiceLoss
from metrics import IOU

def train(model, trainloader, device, num_epoch=5, dataset_desc='unet', lr=0.0001, momentum=0.9,
          savedModel=True):

    # to store loss and accuracy stat
    model_stats = {}

    iou_stats = {
        'train': []
    }

    loss_stats = {
        'train': []
    }

    # compute loss 3 times in each epoch
    loss_iterations = int(np.ceil(len(trainloader) / 3))
    stagnant_count = 0
    best_loss = math.inf

    # transfer model to GPU
    model = model.to(device)

    # set the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # set the scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # init first epoch
    previous_epoch = -1

    # init performance metrics
    loss = BinaryDiceLoss()
    # loss = torch.nn.CrossEntropyLoss()
    iou = IOU()

    # prepare to load checkpoint file
    if savedModel:
        path = './models/' + dataset_desc

        # create models folder if not exists
        if not os.path.isdir('models'):
            os.mkdir('models')

        # load the checkpoint file
        if not os.path.exists(path):
            os.mkdir(path)

        model_files = [f for f in os.listdir(path) if f.endswith('.pt')]
        model_files.sort(reverse=True)

        # load if only exist checkpoint file
        if model_files:
            try:
                checkpoint = torch.load(path + '/' + model_files[0])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                previous_epoch = checkpoint['epoch']
                previous_train_loss = checkpoint['train_loss']

                print(
                    f'Resuming previous epoch. Last run epoch: {previous_epoch}, last train loss: {previous_train_loss}')
            except:
                print("Error in saved model")

                # load the savedModel stat file
        stat_file_name = path + '/' + dataset_desc + '_stat.json'

        if os.path.isfile(stat_file_name):
            with open(stat_file_name, 'r') as f:
                try:
                    model_stats = json.load(f)
                    iou_stats = model_stats["iou"]
                    loss_stats = model_stats["loss"]
                except:
                    print("Error in saved model stat")

    # train the network
    for e in range(previous_epoch + 1, previous_epoch + 1 + num_epoch):

        # set to training mode
        model.train()

        running_train_loss = 0
        running_train_count = 0

        for i, (train_inputs, true_masks) in enumerate(trainloader):
            # Clear all the gradient to 0
            optimizer.zero_grad()

            # transfer data to GPU
            train_inputs = train_inputs.to(device)
            true_masks = true_masks.to(device)

            # forward propagation to get h
            train_outs = model(train_inputs)

            # compute train loss
            train_loss = loss(train_outs, true_masks)

            # compute train iou
            with torch.no_grad():
                preds = torch.sigmoid(train_outs)
                preds = (preds > 0.5).float()
                train_iou = iou(preds, true_masks)

            # backpropagation to get gradients of all parameters
            train_loss.backward()

            # update parameters
            optimizer.step()

            # get the loss
            running_train_loss += train_loss.item()
            running_train_count += 1

            print(f'[Epoch {e + 1:2d} Iter {i + 1:5d}/{len(trainloader)}]: train_loss = {train_loss:.4f}  train_iou: {train_iou:.4f}')

            # display the averaged loss value (training)
            # if i % loss_iterations == loss_iterations - 1 or i == len(trainloader) - 1:
            #     # compute training loss
            #     train_loss = running_train_loss / running_train_count
            #     running_train_loss = 0.
            #     running_train_count = 0.
            #
            #     print(f'[Epoch {e + 1:2d} Iter {i + 1:5d}/{len(trainloader)}]: train_loss = {train_loss:.4f}')

            # track train loss
            loss_stats['train'].append(train_loss.item())

            # track train accuracy
            iou_stats['train'].append(train_iou)

            # save the model stat
            stat_file_name = './models/' + dataset_desc + '/' + dataset_desc + '_stat.json'

            with open(stat_file_name, 'w') as f:
                model_stats["iou"] = iou_stats
                model_stats["loss"] = loss_stats
                json.dump(model_stats, f, indent=2)

        # print train accuracy
        print(f'[Epoch {e + 1:2d}: average train IOU = {train_iou:.2f}')

        # Update the scheduler's counter at the end of each epoch
        scheduler.step()

        # save the model
        if savedModel:
            checkpoint_file = './models/' + dataset_desc + '/' + dataset_desc + '_model_epoch_' + str(e).zfill(
                5) + '.pt'

            torch.save({
                'epoch': e,
                'train_loss': train_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_file)

            # save the model stat
            stat_file_name = './models/' + dataset_desc + '/' + dataset_desc + '_stat.json'

            with open(stat_file_name, 'w') as f:
                model_stats["iou"] = iou_stats
                model_stats["loss"] = loss_stats
                json.dump(model_stats, f, indent=2)

    # return loss_stats, accuracy_stats