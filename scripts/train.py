import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dvs_file_reader import Side
from data_annotation import ObjectID
from sssdata import SSSData, ToTensor
from unet import UNet
from torch.utils.tensorboard import SummaryWriter


def accuracy(label, predicted):
    label_idx = torch.max(label, dim=0)
    predicted_idx = torch.max(predicted, dim=0)


train_dataset = SSSData(
    dvs_filepath='kberg-sidescan/small_farm/data/sss_auto_20210414-085027.dvs',
    annotation_dir='kberg-sidescan/small_farm/object_annotation_corrected',
    side=Side.PORT,
    transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=2)
dim_weights = {
    ObjectID.NADIR.value: .1,
    ObjectID.ROPE.value: 1,
    ObjectID.BUOY.value: 1
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

writer = SummaryWriter()
net = UNet(n_classes=3).to(device)
optimizer = optim.RMSprop(net.parameters())
loss_function = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    losses = []
    running_loss = 0
    for i, sample in enumerate(train_dataloader):
        data, label = sample['data'].to(device), sample['label'].to(device)
        optimizer.zero_grad()

        predicted = net(data)
        loss = loss_function(predicted, label)
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss-batch/train', loss,
                          epoch * len(train_dataloader) + i)

        losses.append(loss.item())
        running_loss += loss.item()
        if i % 50 == 0:
            running_loss /= 50
            print(
                f'[epoch {epoch}, batch {i}] average loss: {running_loss:.5f}')
            running_loss = 0
    average_loss = sum(losses) / len(losses)
    writer.add_scalar('loss-epoch/train', average_loss, epoch)
