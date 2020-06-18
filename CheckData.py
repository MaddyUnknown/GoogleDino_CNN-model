import cv2
import numpy as np

x = np.load(r"Data/2x_new.npy")
y = np.load(r"Data/2_data_y_label.npy")

def save(last=0):
    '''
    Function to check the miss match of the dataset lengths and select the minimum length
    '''
    l = min(len(x), len(y))
    np.save(r"Data/datax_v3.npy", x[0:l-last])
    np.save(r"Data/datay_v3.npy", y[0:l-last])

def display(n=50, late=60):
    '''
    Last n frames are displayed along with the controls recorded with waitKey value of late
    '''
    for i in range(len(x)-n, len(x)):
        cv2.imshow("Display", x[i])
        print(y[i], i)
        if cv2.waitKey(late) & 0xFF == 27:
            break
    print(x.shape, y.shape)
    n = y.tolist()
    print("Up: ",n.count(0))
    print("Stay: ",n.count(1))
    print("Down: ",n.count(2))
    cv2.destroyAllWindows()

print(x.shape, y.shape)
choice =1 # 0 for saving, 1 for showing
if choice== 0:
    save()
elif choice == 1:
    display(len(x), 60)