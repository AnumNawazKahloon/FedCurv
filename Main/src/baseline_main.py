#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tqdm import tqdm
import matplotlib.pyplot as plt
    
import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNPathMnist, CNNCifar

if __name__ == '__main__':
    device = 'cpu'
    args = args_parser()
   
    # Load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # Build model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'pathmnist':
            global_model = CNNPathMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training setup
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []
    epoch_accuracy = []
    test_losses = []
    test_accuracies = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        # Training accuracy and loss
        loss_avg = sum(batch_loss) / len(batch_loss)
        accuracy = 100 * correct / total
        epoch_loss.append(loss_avg)
        epoch_accuracy.append(accuracy)
        print('\nTrain loss:', loss_avg)
        print('Train accuracy: {:.2f}%'.format(accuracy))

        # Testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_losses.append(test_loss)
        test_accuracies.append(100 * test_acc)  # Convert to percentage here
        print('Test loss:', test_loss)
        print('Test accuracy: {:.2f}%'.format(100 * test_acc))

    # Configure fonts first
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'


    # Plotting
    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(epoch_loss, label='Training Loss', color='dodgerblue', linestyle='dashdot',
              linewidth=1, marker = 's', markerfacecolor = 'darkviolet', markersize = 2,
              markeredgecolor="dodgerblue", markeredgewidth=2)
    plt.plot(test_losses, label='Testing Loss', color='teal', linestyle='dashed',
              linewidth=1, marker = 'h', markerfacecolor = 'darkviolet', markersize = 2,
              markeredgecolor="crimson", markeredgewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (%)')
    #plt.ylim(0, 60)  # Set y-axis limits
    #plt.xlim(0, 20)   # Set x-axis limits
    plt.legend()
    #plt.title(f"Combined Loss ({args.dataset}, {args.model}, Epochs={args.epochs})")
    plt.show()

    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(epoch_accuracy, 
             label='Training Accuracy', 
             color='dodgerblue',
             linestyle='dashdot',
             linewidth=1, 
             marker = 's', 
             markerfacecolor = 'darkviolet', 
             markersize = 2,
              markeredgecolor="dodgerblue", 
              markeredgewidth=2)
    plt.plot(test_accuracies, 
             label='Testing Accuracy', 
             color='teal', 
             linestyle='dashed',
              linewidth=1, 
              marker = 'h', 
              markerfacecolor = 'darkviolet', 
              markersize = 2,
              markeredgecolor="crimson", 
              markeredgewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    #plt.ylim(60, 100)  # Set y-axis limits
    #plt.xlim(0, 20)   # Set x-axis limits
    plt.legend()
    #plt.title(f"Combined Accuracy ({args.dataset}, {args.model}, Epochs={args.epochs})")
    plt.show()

    # Final test
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('\nFinal Test Accuracy: {:.2f}%'.format(100 * test_acc))