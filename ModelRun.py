import torch
import torch.nn as nn
import cv2
import pyautogui
import time
import numpy as np
import win32api as wapi

class ToTorch(object):
    def __call__(self, image, i):
        image = image.reshape((1,i,40,140))
        return torch.from_numpy(image)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride =2)
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.batch3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 128, 3)
        self.batch4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(128*1*6, 126)
        self.batch_fc1 = nn.BatchNorm1d(126)
        self.relu_fc1 = nn.ReLU()
        
        self.fc2 = nn.Linear(126, 30)
        self.batch_fc2 = nn.BatchNorm1d(30)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(30, 3)
        
    def forward(self, x, choice=0):
        x = x/255
        x = self.relu1(self.batch1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu2(self.batch2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu3(self.batch3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.relu4(self.batch4(self.conv4(x)))
        x = self.pool4(x)
        
        x = x.view(-1, 128*1*6)
        
        x = self.relu_fc1(self.batch_fc1(self.fc1(x)))
        
        x = self.relu_fc2(self.batch_fc2(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x

def up():
    pyautogui.press('up')

def down():
    pyautogui.press('down')

def snap():
    image = np.array(pyautogui.screenshot(region=(100,205,770,200)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (140,40))
    image = ToTorch()(image, 1)
    return image

def predict():
    paused = False
    image = snap()
    img = image
    image = snap()
    img = np.append(img, image, axis = 1)
    image = snap()
    img = np.append(img, image, axis = 1)

    while True:
        if not paused:
            image = snap()
            img = ToTorch()(np.append(img[0][1:].reshape((1,2,40, 140)), image, axis = 1), 3)
            output = model(img.float())
            output = int(torch.argmax(output, 1))
            #print(output)
            
            if output == 0:
                up()
            elif output == 1:
                pass
            elif output == 2:
                down()
        
        if wapi.GetAsyncKeyState(ord('P')):
            if paused:
                paused = False
                print("Model:  Play", end='\r')
            else:
                paused = True
                print("Model: Pause", end='\r')

        if wapi.GetAsyncKeyState(ord('Q')):
            print("\nThanks for trying...")
            break


model = Net()
model.load_state_dict(torch.load(r'Models\Model_v9\Model_v9'))
model.eval()

for i in range(4):
    print("Strating in: %d sec"%(4-i), end='\r')
    time.sleep(1)

print("Model Playing Google Dino Game....")

predict()
