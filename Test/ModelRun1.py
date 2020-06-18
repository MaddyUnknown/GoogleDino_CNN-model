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
        
        self.conv1_1 = nn.Conv2d(1, 64, 3)
        self.batch1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 128, 3)
        self.batch1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=2)
        
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2_1 = nn.Conv2d(128, 256, 3)
        self.batch2_1 = nn.BatchNorm2d(256)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(256, 256, 3)
        self.batch2_2 = nn.BatchNorm2d(256)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.dropout2 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(256*6*31, 512)
        self.batch_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 126)
        self.batch_fc2 = nn.BatchNorm1d(126)
        self.relu_fc2 = nn.ReLU()
        
        #self.fc3 = nn.Linear(256, 126)
        #self.batch_fc3 = nn.BatchNorm1d(126)
        #self.relu_fc3 = nn.ReLU()
        
        self.fc4 = nn.Linear(126, 3)
        
    def forward(self, x, choice=0):
        x = x/255
        x = self.relu1_1(self.batch1_1(self.conv1_1(x)))
        x = self.relu1_2(self.batch1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        x = self.dropout1(x)
        
        x = self.relu2_1(self.batch2_1(self.conv2_1(x)))
        x = self.relu2_2(self.batch2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        x = self.dropout2(x)
        
        #print(x.shape)
        
        x = x.view(-1, 256*6*31)
        
        x = self.relu_fc1(self.batch_fc1(self.fc1(x)))
        
        x = self.relu_fc2(self.batch_fc2(self.fc2(x)))
        
        #x = self.relu_fc3(self.batch_fc3(self.fc3(x)))
        
        x = self.fc4(x)
        
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
