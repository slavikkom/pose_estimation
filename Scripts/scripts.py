import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed, Convolution2D, Flatten, MaxPooling2D

data = []
y = []
for i1 in range(5):
    for i2 in range(5):
        for i3 in range(5):
            for i4 in range(5):
                for i5 in range(5):
                    for i6 in range(5):
                        im = cv2.imread("C:\\Users\\Slav\\Downloads\\Space_research\\IKD_train_data\\train_data1\\save" +
                                        str(i1) + str(i2)+str(i3)+str(i4) +str(i5)+str(i6) + ".png")
                        data.append(np.reshape(im, (100, 100, 3)))
                        temp = []
                        temp.append(i1)
                        temp.append(i2)
                        temp.append(i3)
                        temp.append(i4)
                        temp.append(i5)
                        temp.append(i6)
                        y.append(temp)
						
# print(len(data))
# print(len(y))
rand_ind = np.random.permutation(len(data))
train_ind = rand_ind[ : int(0.9 * len(data))]
dev_ind = rand_ind[int(0.9 * len(data)) : ]
train_x = np.array([data[k] for k in train_ind])
train_y = np.array([y[k] for k in train_ind])
dev_x = np.array([data[k] for k in dev_ind])
dev_y = np.array([y[k] for k in dev_ind])

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=32, epochs=4, validation_data=[dev_x, dev_y])