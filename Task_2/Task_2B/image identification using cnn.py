import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sys import platform
import os

input_path='D:\PROGRAMS\Eyantra\TASK_2B'
loss_func=nn.CrossEntropyLoss
epochs=10
testing_path='D:\PROGRAMS\Eyantra\TASK_2B'
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transform = {'training': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomGrayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]),
    'testing':transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize])
    }
image_set = {'training': datasets.ImageFolder(input_path + 'training', data_transform["training"])}

data_loader = {'training': torch.utilis.data.DataLoader(
    image_set['training'], batch_size=8, shuffle=True)}

def train(dataloader):
    
train(data_loader)

