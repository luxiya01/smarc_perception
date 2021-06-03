import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dvs_file_reader import Side
from data_annotation import ObjectID
from sssdata import SSSData, ToTensor
from network import Net
from torch.utils.tensorboard import SummaryWriter


def loss_function(label, predicted, weights):
    return (weights * (label - predicted)**2).mean()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

writer = SummaryWriter()
net = Net(num_classes=3).to(device)
lr = .01
weight_decay = .1
batch_size = 16
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
weights = torch.Tensor([.1, .6, 1]).to(device)

train_dataset = SSSData(
    dvs_filepath='kberg-sidescan/small_farm/data/sss_auto_20210414-085027.dvs',
    annotation_dir='kberg-sidescan/small_farm/object_annotation_corrected',
    side=Side.PORT,
    transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)

num_epochs = 10
for epoch in range(num_epochs):
    losses = []
    running_loss = 0
    for i, sample in enumerate(train_dataloader):
        data, label = sample['data'].to(device), sample['label_scaled'].to(
            device)
        optimizer.zero_grad()

        predicted = net(data)
        loss = loss_function(label=label, predicted=predicted, weights=weights)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss-batch/train', loss,
                          epoch * len(train_dataloader) + i)

        losses.append(loss.item())
        running_loss += loss.item()
        if i > 0 and i % 50 == 0:
            running_loss /= 50
            print(
                f'[epoch {epoch}, batch {i}] average loss: {running_loss:.5f}')
            running_loss = 0
    average_loss = sum(losses) / len(losses)
    writer.add_scalar('loss-epoch/train', average_loss, epoch)
    torch.save(
        net.state_dict(),
        f'optim_adam_lr{lr}_weight_decay{weight_decay}_batch{batch_size}_numepochs{num_epochs}_epoch{epoch}.pth'
    )
