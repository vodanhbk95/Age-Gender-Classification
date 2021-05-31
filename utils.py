import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, scheduler, criterion1, criterion2, train_data, phase):
    if phase == "train":
        print('Training')
        model.train()
    else:
        print('Validating')
        model.eval()
    
    train_cls_loss = 0.0
    mae_age_reg_loss = 0.0
    
    age_correct = 0
    gender_correct = 0

    for i, (inputs, age_rgs_label, age_cls_label, gender) in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        inputs, age_rgs_label, age_cls_label, gender = inputs.to(device), age_rgs_label.to(device), age_cls_label.to(device), gender.to(device)

        if phase == "train":
            optimizer.zero_grad()
            gen_pred, age_cls_pred, age_reg_pred = model(inputs)
        else:
            with torch.no_grad():
                gen_pred, age_cls_pred, age_reg_pred = model(inputs)
        
        # if gender.max() > 1 or gender.min() < 0 or age_cls_label.max() >= 10 or age_cls_label.min() < 0:
        #     print(gender.max(),gender.min(),  age_cls_label.min(), age_cls_label.max())
        loss_gender  = criterion1(gen_pred.float(), gender.squeeze().long())
        loss_age_cls = criterion1(age_cls_pred.float(), age_cls_label.squeeze().long())
        loss_age_reg = criterion2(age_reg_pred, age_rgs_label)
        
        # print(loss_gender, loss_age_cls, loss_age_reg)
        sum_cls_loss = loss_age_reg + loss_age_cls + loss_gender
        
        train_cls_loss += sum_cls_loss.item()
        mae_age_reg_loss += loss_age_reg.item()

        _, gender_preds  = torch.max(gen_pred,1)
        _, age_cls_preds = torch.max(age_cls_pred, 1)


        gender_correct += (gender_preds == gender.squeeze()).sum().item()
        age_correct += (age_cls_preds == age_cls_label.squeeze()).sum().item()

        if phase == "train":
            sum_cls_loss.backward()     
            optimizer.step()
    
        

    train_loss = train_cls_loss / (len(dataloader))
    train_mae_age = mae_age_reg_loss / (len(dataloader))

    train_accuracy_gender = 100. * gender_correct / (len(dataloader.dataset))
    train_accuracy_age_cls = 100. * age_correct / (len(dataloader.dataset))

    return train_loss, train_mae_age, train_accuracy_gender, train_accuracy_age_cls

