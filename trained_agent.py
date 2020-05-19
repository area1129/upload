import numpy as np
import keras

def read_data(filename='dataset.txt'):
    f = open(filename,'r')

    boards = []
    directions = []
    
    # read data
    for j in range(150000):
        num = f.readline()
        if not num: # end of file
            break
        num = float(num)
        board = np.zeros((4,4,12))
        for p in range(4):
            for q in range(4):
                if num == 0:
                    board[p, q, 0] = 1
                else:
                    board[p, q, int(np.log2(num))] = 1
                num = float(f.readline())
        boards.append(board)                                # save the board
        direction = int(num)
        directions.append(direction)                        # save the direction
    # convert to numpy array
    boards = np.array(boards)
    directions= np.array(directions)
    # convert to one-hot encoding
    directions = keras.utils.to_categorical(directions, num_classes=4)
    f.close()
    return boards, directions

(boards,directions)=read_data()

# split training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boards, directions, test_size=0.2, random_state=42)

# model
from keras import models
from keras import layers
from keras import optimizers

NUM_EPOCHS = 10
NUM_CLASSES = 4                                             # four directions
BATCH_SIZE = 64
INPUT_SHAPE = (4, 4, 12)                                    # from 0 to 2048

model = models.Sequential()
model.add(layers.Conv2D(64,(2,2),activation='relu',input_shape=INPUT_SHAPE))
model.add(layers.Conv2D(128,(2,2),activation='relu'))
#model.add(layers.Conv2D(128,(2,2),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(NUM_CLASSES,activation='softmax'))

model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,validation_split=0.2)

results = model.evaluate(x_test,y_test)
print(results)

model.save('model051901.h5')