import os
import torch
import torch.nn as nn 
import pandas as pd 
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from dataloader import AgeGender
from utils import train, validate
from model import AgeGenderModel

train_csv = 'process_imdb-wiki/train.csv'
test_csv = 'process_imdb-wiki/test.csv'

# define transforms
transform_train = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# Parameter
batch_size = 15
epochs = 50
lr = 1e-3
num_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AgeGenderModel()
model = nn.DataParallel(model)
model.to(device)
train_data = AgeGender(train_csv, transform=transform_train)
valid_data = AgeGender(test_csv, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15, 30, 50], gamma=0.1)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.L1Loss()

for epoch in range(epochs):
    epoch_start = epoch + 1
    print(f'Epoch {epoch_start} of {epochs}')
    
    train_loss_gender, train_loss_age_cls, train_loss_age_reg, train_accuracy_gender, train_accuracy_age_cls = train(model, train_loader, optimizer, scheduler, criterion1, criterion2, train_data)
    val_loss_gender, val_loss_age_cls, val_loss_age_reg, val_accuracy_gender, val_accuracy_age_cls = validate(model, valid_loader, criterion1, criterion2, val_data)
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    #save checkpoint
    if epoch_start % 5 == 0:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss1': criterion1,
            'loss2': criterion2,
            }, './outputs/model_epoch_{}.pth'.format(epoch_start)
        )
