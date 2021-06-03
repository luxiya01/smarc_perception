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


#TODO: compute per object class accuracy
def compute_accuracy(label, predicted):
    label_value, label_idx = torch.max(label, dim=2)
    predicted_value, predicted_idx = torch.max(predicted, dim=2)

    valid_labels = (label_value > 0)
    num_labels = valid_labels.float().sum()
    num_correct = torch.logical_and(label_idx == predicted_idx,
                                    valid_labels).float().sum()
    print(f'num_labels: {num_labels}, num_correct: {num_correct}')
    print(f'label_idx: {label_idx}')
    print(f'predicted_idx: {predicted_idx}')
    print(f'predicted_value: {predicted_value}')
    print(f'predicted: {predicted[:, :, 10]}')
    return num_correct / num_labels


train_dataset = SSSData(
    dvs_filepath='kberg-sidescan/small_farm/data/sss_auto_20210414-085027.dvs',
    annotation_dir='kberg-sidescan/small_farm/object_annotation_corrected',
    side=Side.PORT,
    transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(train_dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

writer = SummaryWriter()
net = UNet(n_classes=1).to(device)
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
            accuracy = compute_accuracy(label, predicted)
            running_loss /= 50
            print(
                f'[epoch {epoch}, batch {i}] average loss: {running_loss:.5f}, current accuracy: {accuracy}'
            )
            running_loss = 0
    average_loss = sum(losses) / len(losses)
    writer.add_scalar('loss-epoch/train', average_loss, epoch)
