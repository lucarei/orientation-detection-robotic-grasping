import time

import torch
import numpy as np
import torchvision
from torchvision import models, transforms

import cv2 as cv

preprocess = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.resnet18(pretrained=True) # Load net
model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True) # Set final layer to predict one value
net = model.to(device) # Assign net to gpu or cpu
#
net.load_state_dict(torch.load("hello_more_data.torch")) # Load trained model

image = cv.imread("bt9.jpg")

cv.imshow('frame', image)

# convert opencv output from BGR to RGB
image = image[:, :, [2, 1, 0]]
permuted = image

# preprocess
input_tensor = preprocess(image)
input_tensor=input_tensor.to(device=device)


# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# run model
output = net(input_batch)
# do something with output ...

print(output)