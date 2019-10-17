#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models
import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import pandas as pd
from torch.autograd import Variable


# In[2]:


TRANSFORM = transforms.Compose([transforms.CenterCrop(200),#從圖片中心剪裁一個size大小的圖片
                                transforms.RandomHorizontalFlip(0.3),#給定概率隨機的對圖片進行水平映象
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                               ])

TRANSFORM_TEST = transforms.Compose([transforms.CenterCrop(200),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                    ])


INPUT_PATH = r'C:\Users\chyang\Desktop\深度學習影像辨識\dataset\dataset\train'
TESTING_PATH = r'C:\Users\chyang\Desktop\深度學習影像辨識\dataset\dataset\test'

BATCH_SIZE = 5
FIRST_EPOCH = 5
SECOND_EPOCH = 5
FIRST_LEARNING_RATE = 0.001
SECOND_LEARNING_RATE = 0.0001
MOMENTUM = 0.9

category = {0:'bedroom',
            1:'coast',
            2:'forest',
            3:'highway',
            4:'insidecity',
            5:'kitchen',
            6:'livingroom',
            7:'mountain',
            8:'office',
            9:'opencountry',
            10:'street',
            11:'suburb',
            12:'tallbuilding'}

img_data = torchvision.datasets.ImageFolder(INPUT_PATH, transform=TRANSFORM)
data_loader = torch.utils.data.DataLoader(img_data, batch_size=BATCH_SIZE, shuffle=True)

net = models.resnext101_32x8d(pretrained=True)
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=FIRST_LEARNING_RATE, momentum=MOMENTUM)


for epoch in range(FIRST_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

            
optimizer = optim.SGD(net.parameters(), lr=SECOND_LEARNING_RATE, momentum=MOMENTUM)


for epoch in range(SECOND_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

            
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image).float()
    #image = torch.tensor(image, requires_grad=True)
    image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


predict_df = pd.read_csv('sameple_submission_open.csv', index_col = False)

net_eval = net.eval()

for i in range(1040):
    image_path = TESTING_PATH + '\image_' + '{:04d}'.format(i) + '.jpg'
    
    img = image_loader(loader=TRANSFORM_TEST, image_name=image_path)
    var_image = Variable(img).cuda()
    
    output = net_eval(var_image)
    prediction = int(torch.max(output.data.cpu(), 1)[1].numpy())
    predict_df.loc[i,'label'] = category[prediction]

predict_df.to_csv('predict_resnext101_32x8d_pretrain_201_5_5_random(0.3).csv', index = False)


# In[ ]:




