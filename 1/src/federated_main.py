#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNPathMnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'
    device = 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'pathmnist':
            global_model = CNNPathMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

# Training
train_loss, train_accuracy = [], []
test_losses, test_accuracies = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 2
val_loss_pre, counter = 0, 0

for epoch in tqdm(range(args.epochs)):
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # Local training
    for idx in idxs_users:
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
        w, loss = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))

    # Weight averaging
    global_weights = average_weights(local_weights)
    global_model.load_state_dict(global_weights)

    # Test inference
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)

    # Calculate training metrics
    train_loss.append(np.mean(local_losses))
        
    # Training accuracy calculation
    train_accuracies = []
    global_model.eval()
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                idxs=user_groups[c], logger=logger)
        acc, _ = local_model.inference(model=global_model)
        train_accuracies.append(acc)
    train_accuracy.append(np.mean(train_accuracies))

    # Print statistics
    if (epoch+1) % print_every == 0:
        print(f'\nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss: {train_loss[-1]:.4f}')
        print(f'Train Accuracy: {100*train_accuracy[-1]:.2f}%')
        print(f'Test Accuracy: {100*test_accuracies[-1]:.2f}%')

    # [Rest of the code unchanged until plotting section]

    # Enhanced Plotting Section
import matplotlib.pyplot as plt
plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'figure.autolayout': True
})

    # Accuracy Plot
plt.figure(figsize=(3,3), dpi=300)
epochs = range(1, len(train_accuracy)+1)

plt.plot(epochs, [acc*100 for acc in test_accuracies], 
            label='Training Accuracy',
            color='dodgerblue',
            linestyle='dashdot',
            linewidth=1,
            marker='s',
            markerfacecolor='darkviolet',
            markersize=2,
            markeredgecolor='dodgerblue',
            markeredgewidth=2)

plt.plot(epochs, [acc*100 for acc in train_accuracy], 
             label='Test Accuracy',
            color='teal',
            linestyle='dashed',
            linewidth=1,
            marker='h',
            markerfacecolor='darkviolet',
            markersize=2,
            markeredgecolor='crimson',
            markeredgewidth=2)

plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy (%)')
#plt.ylim(0, 100)  # Set y-axis limits
#plt.xlim(0, 20)   # Set x-axis limits
plt.legend()
plt.show()
#plt.savefig('ddv.png', bbox_inches='tight')

    # Loss Plot
plt.figure(figsize=(4, 3.5), dpi=300)
    
plt.plot(epochs, train_loss, 
            label='Training Loss', 
            color='forestgreen',
            linestyle=':',
            linewidth=1.5,
            marker='^',
            markersize=4,
            markeredgecolor='darkgreen',
            markerfacecolor='limegreen')
    
plt.plot(epochs, test_losses, 
            label='Testing Loss', 
            color='darkorange',
            linestyle='--',
            linewidth=1.5,
            marker='D',
            markersize=4,
            markeredgecolor='chocolate',
            markerfacecolor='gold')
    
plt.xlabel('Communication Rounds', fontsize=10)
plt.ylabel('Loss Value', fontsize=10)
plt.legend(frameon=False, fontsize=8)
plt.show()
plt.savefig('xcv.png')
plt.close()