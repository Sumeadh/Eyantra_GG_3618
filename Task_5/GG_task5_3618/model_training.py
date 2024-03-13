import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os
import torch.optim.lr_scheduler as lr_scheduler

out_features = 5

input_path = "./data/"
loss_fn = nn.CrossEntropyLoss()
epochs = 3


testing_location = "./data/test"
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
none = "None"




normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
data_transform = {
    'training': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine((0, 90)), 
        transforms.GaussianBlur((3, 3), (1.0, 2.0)),      
        transforms.ToTensor(),        
        normalize]
    ),
    'testing': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize]
    )
}

image_set = {
    'training': datasets.ImageFolder(input_path + 'training', data_transform["training"])    
}
dataloader = {
    'training': torch.utils.data.DataLoader(image_set['training'], batch_size = 16, shuffle = True)
}

"""def accuracy_prediction(outputs, labels):
    for i in range(len(outputs)):
        _, predicton = torch.max(outputs[i], 1)        
        if predicton == labels[i]:
            acc = acc + 1 
    
    accuracy = (acc/len(outputs))*100
    return accuracy"""


def train(dataloader):
    weights = models.ResNet152_Weights.DEFAULT
    model = models.resnet152(weights = weights)
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
               nn.Linear(2048, 512),
               nn.ReLU(inplace=True),
               nn.Linear(512, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, out_features=out_features))
    
    optimizer = optim.SGD(model.fc.parameters(), lr = 0.05)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor= 1.0, end_factor=0.5, total_iters=3)
    for epoch in range(epochs):
        iter = 0
        acc = 0
        for inputs, labels in dataloader['training']:            
            outputs = model(inputs)
            
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)   
            iter = iter + 1                     
            for i in range(len(preds)):
                if preds[i] == labels[i]:
                    acc = acc +1
        print(acc, iter)
        cal_acc = (acc/(iter*8))*100
        print("Epoch - %d ------------ Accuracy - %d"%(epoch, cal_acc))
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        
    return model

def detect(label):
    if label == 0:
        return combat
    if label == 1:
        return destroyed_building
    if label == 2:
        return fire
    if label == 3:
        return rehab
    if label == 4:
        return military_vehicles

trained_model = train(dataloader)
torch.save(trained_model.state_dict(), 'weights_path_name.pth') 
torch.save(trained_model, 'model5.pth')
Files = []
for root, dir_name, file_name in os.walk(testing_location):
    for name in file_name:
        full_name = os.path.join(root, name)
        Files.append(full_name)
trained_model.eval()
with torch.inference_mode():
    for file in Files:
        img = Image.open(file)
        img = data_transform["testing"](img)
        img = img.unsqueeze(dim = 0)
        logit = trained_model(img)
        _, preds = torch.max(logit, 1)
        event = detect(preds)
        print(file)
        print(event)