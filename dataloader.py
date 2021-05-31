import os
import torch
import csv
import cv2
import numpy as np
import pandas as pd

import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
from torchvision.transforms import transforms

import warnings
warnings.filterwarnings("ignore")

class AgeGender(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        
        img_name = self.csv_file.iloc[idx, 0]
        # print(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
 
        age = self.csv_file['age'].iloc[idx]
        age_rgs_label = torch.from_numpy(np.array([age/100], dtype='float'))

        gender = self.csv_file['gender'].iloc[idx]
        gender = float(gender)
        gender = torch.from_numpy(np.array([gender], dtype='float'))
        gender = gender.type(torch.LongTensor)

        age_cls_label = torch.from_numpy(np.array([int(age/10)], dtype='float'))
        age_cls_label = age_cls_label.type(torch.FloatTensor)
        # print(img_name, gender, age_cls_label)
        # if gender.max() > 1 or gender.min() < 0 or age_cls_label.max() >= 10 or age_cls_label.min() < 0:
        #     print(img_name)

        return image, age_rgs_label, age_cls_label, gender
    
if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])
    
    agegender_dataset = AgeGender(csv_file='process_imdb-wiki/test.csv', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(agegender_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

    from tqdm import tqdm
    for (image, age_rgs_label, age_cls_label, gender) in tqdm(train_loader):
        a=1