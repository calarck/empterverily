import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import requests
import zipfile
import torch
import torchvision.utils
from torchvision import transforms,datasets
import torchvision.models.quantization as models
from torch import nn,optim
from torch.quantization import convert
from PIL import Image,ImageDraw
import cv2

def train_model(model,criterion,optimizer,scheduler,num_epochs = 25,device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc =0.0
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epochs-1))
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs,lables in dataloaders[phase]:
                inputs = inputs.to(device)
                lables = lables.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,lables)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == lables.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} loss {:4f} acc {:4f}".format(phase,epoch_loss,epoch_acc))

            if  phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since

    print("训练耗时 {:0f}m,{:0f}s".format(time_elapsed // 60, time_elapsed % 60))

    print("最好成绩{:4f}", best_acc)

    model.load_state_dict(best_model_wts)

    return model


def create_combined_model(model_fe):
    model_de_features = nn.Sequential(model_fe.quant
                                      ,model_fe.conv1
                                      ,model_fe.bn1
                                      ,model_fe.relu
                                      ,model_fe.maxpool
                                      ,model_fe.layer1
                                      ,model_fe.layer2
                                      ,model_fe.layer3
                                      ,model_fe.layer4
                                      ,model_fe.avgpool
                                      ,model_fe.dequant,)

    new_hea = nn.Sequential(nn.Dropout(0.5),nn.Linear(num_ftrs,2))
    new_model = nn.Sequential(model_de_features,nn.Flatten(1),new_hea)
    return new_model

if __name__ =="__main__":

    DATA_PATH = os.path.join(os.getcwd(), "../../BaiduNetdiskDownload/images")

    # 第二步
    data_transforms = {'train': transforms.Compose([transforms.Resize(224)
                                                       , transforms.RandomCrop(224)
                                                       , transforms.RandomHorizontalFlip()
                                                       , transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        , "val": transforms.Compose([transforms.Resize(224)
                                        , transforms.RandomCrop(224)
                                        , transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_dir = DATA_PATH
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    # num_workers=8 不使用多线程
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x
                   in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fe = models.resnet18(pretrained=True, progress=True, quantize=False)
    num_ftrs = model_fe.fc.in_features
    new_model = create_combined_model(model_fe)
    new_model = new_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    new_model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device=device)
    # 转化换模型
    new_model.cpu()


    #保持模型 加载的时候还要重新转换
    # torch.save(model_qbuantized_and_trained.state_dict(), "../../BaiduNetdiskDownload/empter_o_dirct.pt")
    # 保持模型
    torch.save(new_model, "empter_o.pt")
