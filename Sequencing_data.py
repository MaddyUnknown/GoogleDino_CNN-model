import numpy as np

x = np.load('datax_v3.npy')
y = np.load('datay_v3.npy')

print(x.shape, y.shape)
x = x.reshape(len(x), 1, 40 , 140)

x_new = np.append(x[:-2], x[1:-1], axis=1)
x_new = np.append(x_new, x[2:], axis=1)

y_new = y[2:]


print(x_new.shape, y_new.shape)

np.save("sequence_datax_v3.npy", x_new)
np.save("sequence_datay_v3.npy", y_new)