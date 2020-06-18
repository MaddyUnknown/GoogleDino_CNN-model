import numpy as np
import random

def equalize_data(x, y):
    up_index = np.where(y==0)[0].tolist()
    stay_index = np.where(y==1)[0].tolist()
    down_index = np.where(y == 2)[0].tolist()

    random.shuffle(up_index)
    random.shuffle(stay_index)
    random.shuffle(down_index)

    l = min(len(up_index), len(down_index), len(stay_index))
    
    up_index = up_index[:l]
    down_index = down_index[:l]
    stay_index = stay_index[:l]

    index = list(up_index)
    index.extend(down_index)
    index.extend(stay_index)

    x_new = x[index]
    y_new = y[index]
    print(x_new.shape, y_new.shape)

    return x_new, y_new

def len_classes(x,y):
    up = np.where(y==0)[0]
    stay = np.where(y==1)[0]
    down = np.where(y == 2)[0]

    print("Up data  : ", len(up), end='    ')
    print("Stay data: ", len(stay), end='    ')
    print("Down data: ", len(down))
    print("Total Data :", len(y))

def train_data():
    print("\n\nTrain set: ")
    x1 = np.load('sequence_datax_v1.npy')
    y1 = np.load('sequence_datay_v1.npy')

    x2 = np.load('sequence_datax_v2.npy')
    y2 = np.load('sequence_datay_v2.npy')

    x3 = np.load('sequence_datax_v3.npy')
    y3 = np.load('sequence_datay_v3.npy')

    len_classes(x1, y1)
    new_x1, new_y1 = equalize_data(x1, y1)

    len_classes(x2, y2)
    new_x2, new_y2 = equalize_data(x2, y2)

    len_classes(x3, y3)
    new_x3, new_y3 = equalize_data(x3, y3)

    new_x = np.append(new_x1, new_x2, axis = 0)
    new_y = np.append(new_y1, new_y2, axis = 0)

    new_x = np.append(new_x, new_x3, axis = 0)
    new_y = np.append(new_y, new_y3, axis = 0)

    print("Total data generated: ", len(new_x))

    np.save("Train_set_x_sequence.npy", new_x)
    np.save("Train_set_y_sequence.npy", new_y)

def val_data():
    print("\n\nVal set: ")
    x = np.load('sequence_datax_v4.npy')
    y = np.load('sequence_datay_v4.npy')

    len_classes(x, y)
    new_x, new_y = equalize_data(x, y)

    print("Total data generated: ", len(new_x))

    print("Total data generated: ", len(new_x))

    np.save("Val_set_x_sequence.npy", new_x)
    np.save("Val_set_y_sequence.npy", new_y)

train_data()
val_data()