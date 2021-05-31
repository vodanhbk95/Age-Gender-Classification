import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class AgeGenderModel(torch.nn.Module,):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.resNet = models.resnet18(pretrained=True)

        self.fc1 = nn.Linear(512, 512)
        self.age_cls_pred = nn.Linear(512, 10)
        self.age_reg_pred = nn.Linear(10,1)

        self.fc2 = nn.Linear(512, 512)
        self.gen_cls_pred = nn.Linear(512, 2)
        
        self.dropout = nn.Dropout(0.5)

    def get_resnet_convs_out(self, x):

        x = self.resNet.conv1(x)
        x = self.resNet.bn1(x)
        x = self.resNet.relu(x)
        x = self.resNet.maxpool(x)

        x = self.resNet.layer1(x)
        x = self.resNet.layer2(x)
        x = self.resNet.layer3(x)
        x = self.resNet.layer4(x)

        return x

    def get_age_gender(self, last_conv_out):
        
        last_conv_out = self.resNet.avgpool(last_conv_out)
        last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)
        last_conv_out = self.dropout(last_conv_out)
        
        age_pred = F.relu(self.fc1(last_conv_out))
        age_cls_pred = self.age_cls_pred(age_pred)
        age_reg_pred = self.age_reg_pred(age_cls_pred)
        
        gen_pred = F.relu(self.fc2(last_conv_out))
        gen_pred = self.gen_cls_pred(gen_pred)

        return gen_pred, age_cls_pred, age_reg_pred

    def forward(self, x):
        last1 = self.get_resnet_convs_out(x)
        gen_pred, age_cls_pred, age_reg_pred = self.get_age_gender(last1)
        return gen_pred, age_cls_pred, age_reg_pred


if __name__ == '__main__':
    a = AgeGenderModel()
    x = torch.zeros((1,3,112,112))
    out = a(x)
    print('All good')
    pass