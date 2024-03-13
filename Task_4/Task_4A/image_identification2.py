import cv2
import numpy as np
import os

import torch
#import torch.nn as nn
#import torchvision
from torchvision import models, transforms, datasets
#import torch.optim as optim
#import torch.optim as optim
#import matplotlib.pyplot as plt
from PIL import Image
temp=[]
img_coordinates=[]
img_distinct=[]
event_list={}
def nothing(x):
   pass

def mse(x,y):
    for i in img_distinct:
        print(((x-i[0])**2+(y-i[1])**2)**0.5)
        if ((x-i[0])**2+(y-i[1])**2)**0.5<10:
            print(img_distinct)
            cv2.circle(img, (x,y), 1, (255,0,0), -1)
            return False
    else:
        cv2.circle(img, (x,y), 1, (255,0,0), -1)
        img_distinct.append((x,y))
       # print(img_distinct) 
        
        return True
 
    # Create a directory to save the extracted images
output_directory ='D:\PROGRAMS\Eyantra\Task_4a\extracted_image'
os.makedirs(output_directory, exist_ok=True)
upper=254
lower= 110
u=80
l=60
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
#cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
#cv2.namedWindow('bounded_image', cv2.WINDOW_NORMAL)
#cv2.createTrackbar('upper','thresh',0,500,nothing)
#cv2.createTrackbar('lower','thresh',0,255,nothing)
#cv2.createTrackbar('u','bounded_image',0,500,nothing)
#cv2.createTrackbar('l','bounded_image',0,500,nothing)

r, img = cap.read()
img=img[:, 400:1000, :]
#img=img[:,400:1600,:]
cnt=1
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#image filtering
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply a binary threshold to identify white regions
_, binary = cv2.threshold(blurred, lower, upper, cv2.THRESH_BINARY)
#_, binary = cv2.threshold(blurred, lower, upper, cv2.THRESH_BINARY+ cv2.THRESH_OTSU,) 
#binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)

#upper= cv2.getTrackbarPos('upper','thresh')
#lower= cv2.getTrackbarPos('lower','thresh') 
cv2.imshow('thresh',binary)
contours, hierarchy  = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# to store image of found events

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.rectangle(img,(x,y), (x+w,y+h), (255,150,0), 1)
    # Adjust the conditions based on your selection criteria
    if u*u>w*h>l*l:
        #cv2.drawContours(img, [contour], -1, (0,255,0), 2)         
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Approximate the contour with a polygon
        #cv2.drawContours(img, [approx], -1, (255,0,0), 2) 
        # Draw the smoothed contour on the mask
        if len(approx)==4:
            extracted_img=img[y:y+h,x:x+w]
            Mse=mse(x,y)
            if Mse:
                output_path = os.path.join(output_directory, f'extracted_image_{cnt}.jpg')
                cv2.imwrite(output_path, extracted_img)
                event_list[cnt]=output_path
                cnt+=1
            cv2.drawContours(img, [approx], -1, (255,0,255), 1)     
        cv2.imshow('bounded_image',img)
        #u= cv2.getTrackbarPos('u','bounded_image')
        #l= cv2.getTrackbarPos('l','bounded_image') 
        cv2.waitKey(100)
#Feeding it into ML Model 
trained_model = torch.load('model2.pth')
trained_model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
data_transform = {
    'testing': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize]
    )
}

def detect(label):
    if label == 0:
        return 'combat'
    if label == 1:
        return 'destroyed_building'
    if label == 2:
        return 'fire'
    if label == 3:
        return 'rehab'
    if label == 4:
        return 'military_vehicles'
    
with torch.inference_mode():
    for image in event_list.values():
        image = Image.open(image)
        image = data_transform["testing"](image)
        image = image.unsqueeze(dim = 0)
        logit = trained_model(image)
        _, preds = torch.max(logit, 1)
        event = detect(preds)
        print(event) 


