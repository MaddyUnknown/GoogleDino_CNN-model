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

# Model Structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride =2)
        
        self.dropout1 = nn.Dropout(0.2) 
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.dropout2 = nn.Dropout(0.25) 
        
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.batch3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.dropout3 = nn.Dropout(0.25) 
        
        
        self.fc1 = nn.Linear(64*3*15, 126)
        self.batch_fc1 = nn.BatchNorm1d(126)
        self.relu_fc1 = nn.ReLU()
        
        self.dropout4 = nn.Dropout(0.25) 
        
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
        
        x = x.view(-1, 64*3*15)
        
        x = self.relu_fc1(self.batch_fc1(self.fc1(x)))
        
        x = self.dropout4(x)
        
        x = self.relu_fc2(self.batch_fc2(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x

def up():
    pyautogui.press('up')

def down():
    pyautogui.keyDown('down')
    time.sleep(0.5);
    pyautogui.keyUp('down')


def snap():
    '''
    Change the value of region depending on the position of chrome window, (x,y,h,w)
    '''
    image = np.array(pyautogui.screenshot(region=(170,165,620,160)))    # dpi now set to 100, (secific to my pc) inital 125 dpi corrospont to (100,205,770,200)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("check", image)
    # cv2.waitKey(0)
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
            
            if output == 0:
                up()
                print("Up")
            elif output == 1:
                pass
            elif output == 2:
                down()
                print("Down")
        
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


model = torch.load(r'Model\Model_v11\Model_v11_no_class')
model.eval()

for i in range(4):
    print("Strating in: %d sec"%(4-i), end='\r')
    time.sleep(1)

print("Model Playing Google Dino Game....")

predict()
