import torch

import torch.nn as nn 
from collections import OrderedDict

from model_irse import IR_50
import torch.nn.functional as F

class MixModel(torch.nn.Module):
    def __init__(self):
        super(MixModel, self).__init__()
        self.model_irse = IR_50([112,112])
        
        self.fc1 = nn.Linear(512, 512)
        self.age_cls_pred = nn.Linear(512, 10)
        self.age_reg_pred = nn.Linear(10, 1)
        
        self.fc2 = nn.Linear(512, 512)
        self.gen_cls_pred = nn.Linear(512, 2)
        
        # self.dropout = nn.Dropout(0.5)
        
    def get_irse_convs_out(self, x):
        x = self.model_irse.input_layer(x)
        x = self.model_irse.body(x)
        
        x = self.model_irse.output_layer(x)
        
        return x
    
    def get_age_gender(self, last_conv_out):
        # import ipdb; ipdb.set_trace()
        # # m = nn.AdaptiveAvgPool2d((1, 1))
        # import ipdb; ipdb.set_trace()
        # last_conv_out = m(last_conv_out)
        
        # last_conv_out = last_conv_out.view(last_conv_out.size(0), -1)
        # last_conv_out = self.dropout(last_conv_out)
        
        age_pred = F.relu(self.fc1(last_conv_out))
        age_cls_pred = self.age_cls_pred(age_pred)
        age_reg_pred = self.age_reg_pred(age_cls_pred)
        
        gen_pred = F.relu(self.fc2(last_conv_out))
        gen_pred = self.gen_cls_pred(gen_pred)

        return gen_pred, age_cls_pred, age_reg_pred
        
    def forward(self, x):
        last1 = self.get_irse_convs_out(x)

        gen_pred, age_cls_pred, age_reg_pred = self.get_age_gender(last1)
        return gen_pred, age_cls_pred, age_reg_pred

if __name__ == '__main__':
    a = MixModel()
    model = IR_50([112,112])
    ckpt = torch.load('IR_50_E15.pth')
    model.load_state_dict(ckpt)
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = "model_irse."+k
        new_state_dict[name] = v
    a.load_state_dict(new_state_dict, strict=False)
    import ipdb; ipdb.set_trace()
    x = torch.zeros((2, 3, 112, 112))
    out = a(x)
    print('All good')
    pass