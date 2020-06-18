import pyautogui as pag
import cv2
import numpy as np
import win32api as wapi
import time

keyList = [38,40]                  # 38 is the code for up arrow key and 40 is the key for down arrow key

def key_check():
    '''
    Checks the keys pressed
    '''
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(key)
    return keys

def one_hot(l):
    '''
    Convert the pressed keys into one hot encoding 
    [0,0,0] No action
    [1,0,0] Up arrow key pressed
    [0,0,1] Down arrow key pressed
    '''
    output = [0,0,0] # up, hold, down
    if 38 in l:
        output[0] = 1
    elif 40 in l:
        output[2] = 1
    else:
        output[1] = 1
    return output

def label(l):
    '''
    Convert the pressed keys into one hot encoding 
    [0,0,0] No action
    [1,0,0] Up arrow key pressed
    [0,0,1] Down arrow key pressed
    '''
    if 38 in l:
        return 0
    elif 40 in l:
        return 2
    else:
        return 1

def record():
    '''
    PyAutoGUI to grab the screen of a specified coordinate, and win32api for the key press
    Appends new data to the specified file
    '''
    try:
        data_x = np.load("Data/datax_v3.npy")
        data_y = np.load("Data/datay_v3.npy")
    except:
        data_x = np.array([])
        data_y = np.array([])
    data_x_temp = np.array([])
    data_y_temp = np.array([])
    i=4
    while i>0:
        print(i)
        time.sleep(1)
        i-=1
    print("Starting recording!!.....")
    while True:
        try:
            img = np.array(pag.screenshot(region=(100,205,770,200)))
            #cv2.imshow("Window", img)
            output = np.array(label(key_check()))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (140,40))
            
            if len(data_x_temp) == 0:
                #data_x_temp = img.reshape((1, 200, 770, 3))
                data_x_temp = img.reshape((1, 40, 140, 1))
                data_y_temp = output.reshape((1))
            else:
                #data_x_temp = np.append(data_x_temp , img.reshape((1,200, 770, 3)) ,axis=0)
                data_x_temp = np.append(data_x_temp , img.reshape((1,40, 140, 1)) ,axis=0)
                data_y_temp = np.append(data_y_temp, output.reshape((1)), axis=0)
            if(i%100 == 99):
                print("More data added, Shape of data: ", end='')
                if len(data_x) == 0:
                    data_x = data_x_temp
                    data_y = data_y_temp
                else:
                    data_x = np.append(data_x , data_x_temp ,axis=0)
                    data_y = np.append(data_y, data_y_temp, axis=0)
                np.save(r'Data/datax_v3.npy', data_x)
                np.save(r'Data/datay_v3.npy', data_y)
                data_x_temp = np.array([])
                data_y_temp = np.array([])
                print("Data x: ",data_x.shape,"Data y: ",data_y.shape)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            i+=1
        except KeyboardInterrupt:
            break

    #cv2.destroyAllWindows()

if __name__ == "__main__":
    record()
