import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, scheduler, criterion1, criterion2, train_data):
    print('Training')
    model.train()

    mae_age_reg_loss = 0.0
    age_cls_loss = 0.0
    gender_cls_loss = 0.0

    age_correct = 0
    gender_correct = 0

    for i, (inputs, age_rgs_label, age_cls_label, gender) in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        inputs, age_rgs_label, age_cls_label, gender = inputs.to(device), age_rgs_label.to(device), age_cls_label.to(device), gender.to(device)

        optimizer.zero_grad()
        
        gen_pred, age_cls_pred, age_reg_pred = model(inputs)
        
        loss_gender  = criterion1(gen_pred, gender)
        loss_age_cls = criterion1(age_cls_pred, age_cls_label)
        loss_age_reg = criterion2(age_reg_pred, age_rgs_label)

        gender_cls_loss += loss_gender.item()
        age_cls_loss += loss_age_cls.item()
        mae_age_reg_loss += loss_age_reg.item()

        _, gender_preds  = torch.max(gen_pred,1)
        _, age_cls_preds = torch.max(age_cls, 1)

        gender_correct += (gender_preds == gender).sum().item()
        age_correct += (age_cls_preds == age_cls_label).sum().item()

        loss_gender.backward()
        loss_age_cls.backward()
        loss_age_reg.backward()

        optimizer.step()
        scheduler.step()
    
    train_loss_gender = gender_cls_loss / (i+1)
    train_loss_age_cls = age_cls_loss / (i+1)
    train_mae_age_reg_loss = mae_age_reg_loss / (i+1)

    train_accuracy_gender = 100. * gender_correct / (len(dataloader.dataset))
    train_accuracy_age_cls = 100. * age_correct / (len(dataloader.dataset))
    
    return train_loss_gender, train_loss_age_cls, train_mae_age_reg_loss, train_accuracy_gender, train_accuracy_age_cls

def validate(model, dataloader, criterion1, criterion2, val_data):
    print('Validating')
    model.eval()
    
    mae_age_reg_loss = 0.0
    age_cls_loss = 0.0
    gender_cls_loss = 0.0

    with torch.no_grad():
        for i, (inputs, age_rgs_label, age_cls_label, gender) in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            inputs, age_rgs_label, age_cls_label, gender = inputs.to(device), age_rgs_label.to(device), age_cls_label.to(device), gender.to(device)

            gen_pred, age_cls_pred, age_reg_pred = model(inputs)
        
            loss_gender  = criterion1(gen_pred, gender)
            loss_age_cls = criterion1(age_cls_pred, age_cls_label)
            loss_age_reg = criterion2(age_reg_pred, age_rgs_label)

            gender_cls_loss += loss_gender.item()
            age_cls_loss += loss_age_cls.item()
            mae_age_reg_loss += loss_age_reg.item()

            _, gender_preds  = torch.max(gen_pred,1)
            _, age_cls_preds = torch.max(age_cls, 1)

            gender_correct += (gender_preds == gender).sum().item()
            age_correct += (age_cls_preds == age_cls_label).sum().item()
        
        val_loss_gender = gender_cls_loss / (i+1)
        val_loss_age_cls = age_cls_loss / (i+1)
        valid_mae_age_reg_loss = mae_age_reg_loss / (i+1)

        val_accuracy_gender = 100. * gender_correct / (len(dataloader.dataset))
        val_accuracy_age_cls = 100. * age_correct / (len(dataloader.dataset))

        return val_loss_gender, val_loss_age_cls, valid_mae_age_reg_loss, val_accuracy_gender, val_accuracy_age_cls











