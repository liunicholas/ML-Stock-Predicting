# Python 3.8.6
import time

# tensorflow 2.4.0
# matplotlib 3.3.3
# numpy 1.19.4
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
from yfinance import *

# import sys
#
# terminalOutput = open("terminalOutput.txt", "w")
# sys.stdout = terminalOutput

#TODO: make gpu work??
# devices = tf.config.list_physical_devices('GPU')
# if len(devices) > 0:
#     print('[INFO] GPU is detected.')
# else:
#     print('[INFO] GPU not detected.')

print('[INFO] Done importing packages.')


#TODO: custom keras callback
# Set to zero to use above saved model
TRAIN_EPOCHS = 1
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

    return hist

def getX(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    # open = histNP[0]
    high = histNP[1]
    # low = histNP[2]
    # close = histNP[3]

    X = []
    #19 previous days and predict 20th day
    for x in range(len(high)-20):
        X.append(high[x:x+20])

    X = np.array(X)

    return X

def getY(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    # open = histNP[0]
    high = histNP[1]
    # low = histNP[2]
    # close = histNP[3]

    Y = []
    #19 previous days and predict 20th day
    for y in range(len(high)-20):
        Y.append(high[y+20])

    Y = np.array(Y)
    Y = Y.flatten()

    return Y

# def combineData(histIndex):
#     X = []
#     print(histIndex.shape)
#     for i in range(histIndex.shape[1]):
#         row = []
#         for x in range(histIndex.shape[0]):
#             row.append(histIndex[x][i])
#         X.append(row)
#
#     X = np.array(X)
#
#     return X

#TODO: play with nn architecture later
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

        self.model.add(layers.Conv1D(32, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        # self.model.add(layers.Conv1D(64, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))

        # self.model.add(layers.MaxPooling1D(pool_size = 2))

        self.model.add(layers.Flatten())

        # Now, we flatten to one dimension, so we go to just length 400.
        self.model.add(layers.Dense(2400, activation = 'relu'))
        self.model.add(layers.Dense(1200, activation = 'relu'))
        self.model.add(layers.Dense(600, activation = 'relu'))
        self.model.add(layers.Dense(300, activation = 'relu'))
        self.model.add(layers.Dense(120, activation = 'relu'))
        self.model.add(layers.Dense(60, activation = 'relu'))
        self.model.add(layers.Dense(20, activation = 'relu'))
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

#PLAN: make each row a set of stocks in an index
INDEX_STOCKS = ["AAPL", "MSFT", "AMZN", "FB", "GOOGL", "GOOG", "TSLA", "BRK.B", "JPM", "JNJ"]
INDEX = "SPY"

trainStart = "2014-01-01"
trainEnd = "2019-12-31"

testStart = "2020-01-01"
testEnd = "2020-12-30"

#get all the stock data
stockHistsTrainX = []
stockHistsTestX = []
for stock in INDEX_STOCKS:
    print(f"[INFO] Loading Dataset For {stock}.")
    train = getData(f"{stock}", trainStart, trainEnd)
    test = getData(f"{stock}", testStart, testEnd)
    trainXstock = np.transpose(getX(train))
    testXstock = np.transpose(getX(test))
    # trainXstock = getX(train).reshape()
    # testXstock = getX(test).reshape()
    print(f"stock shape: {trainXstock.shape}")

    if trainXstock.shape[0] != 0:
        trainXstock = trainXstock.reshape((1,20,-1))
        testXstock = testXstock.reshape((1,20,-1))
        trainXstock = np.swapaxes(trainXstock,0,1)
        testXstock = np.swapaxes(testXstock,0,1)
        stockHistsTrainX.append(trainXstock)
        stockHistsTestX.append(testXstock)


print(f"stock shape after reshape: {stockHistsTrainX[0].shape}")
print(stockHistsTrainX[0])

#must figure out how to stack these
stockHistsTrainX = np.vstack(stockHistsTrainX)
stockHistsTestX = np.vstack(stockHistsTestX)
trainX = np.transpose(stockHistsTrainX)
testX = np.transpose(stockHistsTestX)

print(f"total shape: {trainX.shape}")

# stockHistsTrainX = numpy.delete(stockHistsTrainX, 0)
# stockHistsTestX = numpy.delete(stockHistsTestX, 0)

# stockHistsTrainX = np.array(stockHistsTrainX)
# stockHistsTestX = np.array(stockHistsTestX)

#reorganize x data of stocks
# print("[INFO] Combining X Data.")
# trainX = combineData(stockHistsTrainX)
# testX = combineData(stockHistsTestX)
# print(trainX.shape)

#index target prices
print("[INFO] Loading Index Data.")
histTrainIndex = getData(f"{INDEX}", trainStart, trainEnd)
histTestIndex = getData(f"{INDEX}", testStart, testEnd)
trainY = getY(histTrainIndex)
testY = getY(histTestIndex)
# print(f"stock shape: {trainY.shape}")

if TRAIN:
    numStocks = len(stockHistsTrainX)
    net=Net((numStocks, 20))
    print(net)

    results = net.model.fit(generator(BATCH_SIZE_TRAIN, trainX, trainY), validation_data=generator(BATCH_SIZE_TEST, testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1, steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN, validation_steps=len(testX)/BATCH_SIZE_TEST)

    net.model.save("./models")

    predictions = net.model.predict(testX)

    fig = plt.figure("preds vs real high price", figsize=(10, 8))
    fig.tight_layout()
    plt1 = fig.add_subplot(211)
    plt1.title.set_text("training and validation loss")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    plt1.legend(loc='upper right')
    plt2 = fig.add_subplot(212)
    plt2.plot(histTestIndex.index, testY, color="blue", label="real 2020")
    plt2.plot(histTestIndex.index, predictions, color="red", label="preds 2020")
    plt2.legend(loc='upper right')
    plt.savefig("pyplots/newestPlot.png")
    plt.show()

if LOAD:
    oldModel = tf.keras.models.load_model("./models/")
