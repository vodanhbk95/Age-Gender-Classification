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
from skimage import io

import warnings
warnings.filterwarnings("ignore")

class AgeGender(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace()
        img_name = self.csv_file.iloc[idx, 0]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        if len(image.shape) == 2:
            image = torch.stack((image, image, image))
        age = self.csv_file['age'].iloc[idx]
        age_rgs_label = torch.from_numpy(np.array([age/100], dtype='float'))

        gender = self.csv_file['gender'].iloc[idx]
        gender = float(gender)
        gender = torch.from_numpy(np.array([gender], dtype='float'))
        gender.type(torch.LongTensor)

        age_cls_label = torch.from_numpy(np.array([int(age/10)], dtype='float'))
        age_cls_label = age_cls_label.type(torch.FloatTensor)


        return image, age_rgs_label, age_cls_label, gender

if __name__ == "__main__":

    agegender_dataset = AgeGender(csv_file='process_imdb-wiki/train.csv', transform=None)
    train_loader = torch.utils.data.DataLoader(agegender_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    # data = iter(agegender_dataset)

    # for i in range(len(agegender_dataset)):
    #     image, age_rgs_label, age_cls_label, gender = next(data)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     print(image.shape, age_rgs_label, age_cls_label, gender)
    #     cv2.imwrite(f'./sample.jpg', image)
    for (image, age_rgs_label, age_cls_label, gender) in train_loader:
        print()