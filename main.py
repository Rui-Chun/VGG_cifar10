import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


import numpy as np

import os
import argparse
import pickle

from model import VGG



def train(epoch):
    print('\n Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = LossFunc(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            loss_seq.append(train_loss / (batch_idx + 1))
            acc_seq.append((1.0*correct) / total)

        # print('current batch :%d|total:%d|| Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / 500, 100. * correct / total, correct, total))

    # plt.subplot(121)
    # plt.plot(range(1, len(loss_seq)+1), loss_seq, color='red', linewidth=1.0, linestyle='-')
    # plt.title("Loss Curve")
    # plt.subplot(122)
    # plt.plot(range(1, len(acc_seq)+1), acc_seq, color='green', linewidth=1.0, linestyle='-')
    # plt.title("Acc Curve")
    #
    # plt.savefig('./img/result_epoch_'+num2str(epoch)+'.jpg')

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = LossFunc(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print("==========================================================")
    print('Testing Result(Epoch:%d)=======> Acc: %.3f%% (%d/%d)' % (epoch, 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving checkpoint...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 with VGG ')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    config = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    loss_seq = []
    acc_seq = []

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('======> Building model...')
    net = VGG('VGG19')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if config.resume:
        # Load checkpoint.
        print('====> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    LossFunc = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch+50):
        train(epoch)
        test(epoch)
        
        file = open('Loss_seq2.txt', 'wb')
        pickle.dump(loss_seq, file)
        file.close()

        file = open('Acc_seq2.txt', 'wb')
        pickle.dump(acc_seq, file)
        file.close()

