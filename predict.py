# Python 3.8.6
import time

# tensorflow 2.4.0
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
from tensorflow.keras.layers.experimental import preprocessing as preprocessing2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from yfinance import *

# import sys
#
# terminalOutput = open("terminalOutput.txt", "w")
# sys.stdout = terminalOutput

# devices = tf.config.list_physical_devices('GPU')
# if len(devices) > 0:
#     print('[INFO] GPU is detected.')
#     # print('[INFO] GPU is detected.', flush = True)
# else:
#     print('[INFO] GPU not detected.')
#     # print('[INFO] GPU not detected.', flush = True)
# print('[INFO] Done importing packages.')
# # print('[INFO] Done importing packages.', flush = True)

# Set to zero to use above saved model
TRAIN_EPOCHS = 10
# # If you want to save the model at every epoch in a subfolder set to 'True'
# SAVE_EPOCHS = False
# # If you just want to save the final output in current folder, set to 'True
# SAVE_LAST = False
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

TRAIN = True
LOAD = False

def generator(batchSize, x, y):
    index = 0
    while index < len(x):
        batchX, batchY = [], []
        for i in range(batchSize):
            if (index+i)<len(x):
                batchX.append(x[index+i])
                batchY.append(y[index+i])
                index+=1
            else:
                index=0
        yield np.array(batchX), np.array(batchY)

def getData(stockName, startDate, endDate):
    stock = Ticker(stockName)
    hist = stock.history(start=startDate, end=endDate)

    # print(hist)

    # dates = pd.to_datetime(hist.index, format='%Y-%m-%d %H:%M:%S.%f')
    # hist.set_index(dates,inplace=True)

    # print(hist)
    return hist

def displayStock(data, ticker):
    hist = data

    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    open = histNP[0]
    high = histNP[1]
    low = histNP[2]
    close = histNP[3]

    # print(histNP)
    fig = plt.figure(f"{ticker} stock price", figsize=(10, 4))
    plt1 = fig.add_subplot(111)
    plt1.title.set_text("stock price")
    plt1.plot(hist.index, open, color="yellow", label="open")
    plt1.plot(hist.index, high, color="green", label="high")
    plt1.plot(hist.index, low, color="red", label="low")
    plt1.plot(hist.index, close, color="orange", label="close")
    plt1.legend(loc='upper left')
    plt.show()

def getXY(hist):
    histNP = hist.to_numpy()
    #to get columns
    # histNP = np.transpose(histNP)
    # open = histNP[0]
    # high = histNP[1]
    # low = histNP[2]
    # close = histNP[3]

    # histNP = np.transpose(histNP)
    #all columns
    # print(histNP)
    X = np.copy(histNP)
    index = histNP.shape[0]-1
    X = np.delete(X, index, 0)
    # X = np.delete(X, [0,1,2,3], 1)
    # print(X)

    histNP = np.transpose(histNP)
    #just high
    Y = histNP[1]
    Y = np.delete(Y, 0)
    Y = Y.flatten()

    print(X.shape, Y.shape)

    return X, Y

class Net():
    def __init__(self, input_shape):
        # input_shape is assumed to be 4 dimensions: 1. Batch Size, 2. Image Width, 3. Image Height, 4. Number of Channels.
        # You might see this called "channels_last" format.
        self.model = models.Sequential()
        # For the first convolution, you need to give it the input_shape.  Notice that we chop off the batch size in the function.
        # In our example, input_shape is 4 x 32 x 32 x 3.  But then it becomes 32 x 32 x 3, since we've chopped off the batch size.
        # For Conv2D, you give it: Outgoing Layers, Frame size.  Everything else needs a keyword.
        # Popular keyword choices: strides (default is strides=1), padding (="valid" means 0, ="same" means whatever gives same output width/height as input).  Not sure yet what to do if you want some other padding.
        # Activation function is built right into the Conv2D function as a keyword argument.

        self.model.add(layers.Conv1D(8, 3, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        # For MaxPooling2D, default strides is equal to pool_size.  Batch and layers are assumed to match whatever comes in.
        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Conv1D(16, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))
        # # In our example, we are now at 10 x 10 x 16.
        # self.model.add(layers.Conv1D(32, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        #
        # self.model.add(layers.Conv1D(64, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))

        # self.model.add(layers.MaxPooling1D(pool_size = 2))

        self.model.add(layers.Flatten())

        # Now, we flatten to one dimension, so we go to just length 400.
        self.model.add(layers.Dense(2400, activation = 'relu', input_shape = input_shape))
        self.model.add(layers.Dense(1200, activation = 'relu'))
        self.model.add(layers.Dense(600, activation = 'relu'))
        self.model.add(layers.Dense(300, activation = 'relu'))
        self.model.add(layers.Dense(120, activation = 'relu'))
        self.model.add(layers.Dense(60, activation = 'relu'))
        self.model.add(layers.Dense(30, activation = 'relu'))
        self.model.add(layers.Dense(1))
        # Now we're at length 10, which is our number of classes.
        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.Adam(lr=0.001)
        #absolute for regression, squared for classification

        #Absolute for few outliers
        #squared to aggresively diminish outliers
        self.loss = losses.MeanSquaredError()
        #metrics=['accuracy']
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['mse'])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)

print("[INFO] Loading Traning and Test Datasets.")

histTrain = getData("TSLA", "2019-01-01", "2020-01-01")
histTest = getData("TSLA", "2020-01-01", "2021-01-01")

trainX, trainY = getXY(histTrain)
# print(trainX)
testX, testY = getXY(histTest)

# np.transpose(trainX)
# maxes = []
# for x in trainX:
#     maxes.append(np.amax(trainX))
# # np.transpose(trainX)
#
# np.transpose(testX)
# for x in range(12):
#     max = np.amax(testX[x])
#     if max > maxes[x]:
#         maxes[x] = max
# # np.transpose(testX)
#
# for x in range(12):
#     testX[x]/maxes[x]
#     trainX[x]/maxes[x]
#
# print(testX)
# print(trainX)
#
# np.transpose(trainX)
# np.transpose(testX)

#normalization

pt = preprocessing.PowerTransformer()

trainX = np.transpose(trainX)
trainX = pt.fit_transform(trainX)
trainX = np.transpose(trainX)

testX = np.transpose(testX)
testX = pt.fit_transform(testX)
testX = np.transpose(testX)


#trying to use conv 1d
#number of rows, columns/row, 1
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
print(trainX.shape)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
print(testX.shape)

# print(testX)

if TRAIN:
    #this works but need to figure out why
    net=Net((7,1))
    # Notice that this will print both to console and to file.
    print(net)

    results = net.model.fit(generator(BATCH_SIZE_TRAIN, trainX, trainY), validation_data=generator(BATCH_SIZE_TEST, testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1, steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN, validation_steps=len(testX)/BATCH_SIZE_TEST)

    net.model.save("./models")

    # tf.io.write_file("newestRun.txt", results)

    #dont need the batch_size=4
    theModel = net.model.evaluate(testX, testY)

    predictions = net.model.predict(testX).flatten()
    # print(testX.shape)
    # print(predictions.shape)

    fig = plt.figure("preds vs real high price", figsize=(10, 8))
    fig.tight_layout()
    plt1 = fig.add_subplot(211)
    plt1.title.set_text("training and validation loss")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    plt1.legend(loc='upper right')
    plt2 = fig.add_subplot(212)
    histTest = histTest.iloc[1:]
    plt2.plot(histTest.index, testY, color="blue", label="train")
    plt2.plot(histTest.index, predictions, color="red", label="test")
    plt2.legend(loc='upper right')
    plt.savefig("pyplots/newestPlot.png")
    plt.show()

    # terminalOutput.close()

    # print(theModel)
    # print(predictions)

    # # plt.figure()
    # plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'])
    # plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'])
    # plt.show()
    # plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['mae'])
    # plt.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_mae'])
    # plt.show()

if LOAD:
    oldModel = tf.keras.models.load_model("./models/")

    # theModel = oldModel.evaluate(testX, testY)
    # predictions = oldModel.predict(testX).flatten()
    #
    # fig = plt.figure("preds vs real", figsize=(10, 4))
    # fig.tight_layout()
    # plt1 = fig.add_subplot(121)
    # plt1.title.set_text("histogram of preds vs real")
    # plt1.hist2d(testY, predictions, bins=100)
    # plt2 = fig.add_subplot(122)
    # plt2.title.set_text("best fit line of preds vs real")
    # plt2.scatter(testY, predictions)
    # m, b = np.polyfit(testY, predictions, 1)
    # plt2.plot(testY,m*testY+b)
    # # plt3 = fig.add_subplot(223)
    # # plt3.title.set_text("training and validation loss")
    # # plt3.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    # # plt3.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    # # plt3.legend(loc='upper right')
    # # plt4 = fig.add_subplot(224)
    # # plt4.title.set_text("training and validation mse")
    # # plt4.plot(np.arange(0, TRAIN_EPOCHS), results.history['mse'],color="green", label="real")
    # # plt4.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_mse'], color="red", label="preds")
    # # plt4.legend(loc='upper right')
    # plt.savefig("pyplots/newestPlot.png")
    # plt.show()
