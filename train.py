import os
import torch
import torch.nn as nn 
import pandas as pd 
import matplotlib.pyplot as plt
import torch.optim as optim

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

from dataloader import AgeGender
from utils import train
from model import AgeGenderModel
from model_irse import IR_50
from mix_model import MixModel

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

train_csv = 'process_imdb-wiki/train_combinedataset.csv'
test_csv = 'process_imdb-wiki/test_combinedataset.csv'

# define transforms
transform_train = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation((-10,10)),
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
batch_size = 128
epochs = 50
lr = 1e-3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MixModel()
ckpt = torch.load('IR_50_E15.pth')
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    name = "model_irse."+ k
    new_state_dict[name] = v
# model = AgeGenderModel()
model.load_state_dict(new_state_dict, strict=False)
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
    
    if epoch <= 5:
        for name, param in model.named_parameters():
            if param.requires_grad and 'model_irse' in name:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
        
    print(f'Epoch {epoch_start} of {epochs}')
    train_loss, train_mae_age, train_accuracy_gender, train_accuracy_age_cls = train(model, train_loader, optimizer, scheduler, criterion1, criterion2, train_data, phase="train")
    print(f'Loss total {train_loss} | Age mae {train_mae_age} | Gender acc {train_accuracy_gender} | Bin accuracy {train_accuracy_age_cls}')
    valid_loss, valid_mae_age, valid_accuracy_gender, valid_accuracy_age_cls = train(model, valid_loader, optimizer, scheduler, criterion1, criterion2, valid_data, phase="valid")
    print(f'Loss cls {valid_loss} | Age mae {valid_mae_age} | Gender acc {valid_accuracy_gender} | Bin accuracy {valid_accuracy_age_cls}')
    scheduler.step()
    if not os.path.exists('./outputs_w_free'):
        os.makedirs('./outputs_w_free')
    #save checkpoint
    if epoch_start % 5 == 0:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss1': criterion1,
            'loss2': criterion2,
            }, './outputs_w_free/model_epoch_{}.pth'.format(epoch_start)
        )
