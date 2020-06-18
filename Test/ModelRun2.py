import torch
import torch.nn as nn
import cv2
import pyautogui
import time
import numpy as np
import win32api as wapi

class ToTorch(object):
    def __call__(self, image):
        image = image.reshape((1,1,40,140))
        return torch.from_numpy(image)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride =2)
        
        self.dropout1 = nn.Dropout(0.1) 
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.dropout2 = nn.Dropout(0.15) 
        
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.batch3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.dropout3 = nn.Dropout(0.15) 
        
        self.conv4 = nn.Conv2d(32, 128, 3)
        self.batch4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        self.dropout4 = nn.Dropout(0.2) 
        
        self.fc1 = nn.Linear(128*1*6, 126)
        self.batch_fc1 = nn.BatchNorm1d(126)
        self.relu_fc1 = nn.ReLU()
        
        self.dropout5 = nn.Dropout(0.15) 
        
        self.fc2 = nn.Linear(126, 30)
        self.batch_fc2 = nn.BatchNorm1d(30)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(30, 3)
        
    def forward(self, x, choice=0):
        x = x/255
        x = self.relu1(self.batch1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.dropout1(x)
        
        x = self.relu2(self.batch2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.dropout2(x)
        
        x = self.relu3(self.batch3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.dropout3(x)
        
        x = self.relu4(self.batch4(self.conv4(x)))
        x = self.pool4(x)

        x = self.dropout4(x)
        
        #print(x.shape)
        
        x = x.view(-1, 128*1*6)
        
        x = self.relu_fc1(self.batch_fc1(self.fc1(x)))
        
        x = self.dropout5(x)
        
        x = self.relu_fc2(self.batch_fc2(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x

def up():
    pyautogui.press('up')

def down():
    pyautogui.press('down')

def predict():
    paused = False
    while True:
        if not paused:
            image = np.array(pyautogui.screenshot(region=(100,205,770,200)))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (140,40))
            image = ToTorch()(image)
            output = model(image.float())
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
model.load_state_dict(torch.load(r'Models\Model_v3\Model_v3'))
model.eval()

for i in range(4):
    print("Strating in: %d sec"%(4-i), end='\r')
    time.sleep(1)

print("Model Playing Google Dino Game....")

predict()