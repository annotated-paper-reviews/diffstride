import os
import argparse
#from argparse import ArgumentParser
from argparse_dataclass import ArgumentParser
from typing import List
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import ResNet, Bottleneck, BasicBlock

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("*"*50)
print(f'Detected device: {device}')
print("*"*50)

@dataclass
class Arguments:
    """
    Arguments pertaining to training
    """
    batch_size: int = field(default=128,)
    epochs: int = field(default=8,)
    log_step: int = field(default=500,)
    data_path: str = field(default='./data',)
    learning_rate: float = field(default=0.001,)
    momentum: float = field(default=0.9,)
    #num_blocks: List[int] = field(default_factory=[2,2,2,2],)


def evaluate(model, dataloader):
    print("*"*50)
    print(f'Evaluating...')
    print("*"*50)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def main():
    parser = ArgumentParser(Arguments)
    args = parser.parse_args([])
    
    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, 
        train=True, 
        download=True, 
        transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2)
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, 
        train=False, 
        download=True, 
        transform=transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2)

    # model, loss, optimizer
    model = ResNet(BasicBlock, [2,2,2,2]) #resnet18
    model.to(device)
    print("*"*50)
    print('Model initialized')
    print("*"*50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # train
    print("*"*50)
    print(f'Training...')
    print("*"*50)
    step = 0
    for epoch in range(args.epochs):
        
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % args.log_step == args.log_step - 1:
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0       
            step += 1
    
    evaluate(model, test_dataloader)
        

if __name__ == '__main__':
    main()
