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
# import tensorflow.keras.preprocessing as preprocessing
from tensorflow.keras.layers.experimental import preprocessing as preprocessing2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from yfinance import *

# import sys
#
# terminalOutput = open("terminalOutput.txt", "w")
# sys.stdout = terminalOutput

# fout = open('test.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
# print("[TIMER] Process Time:", now, flush = True)

# File location to save to or load from
# MODEL_SAVE_PATH = './boston.pth'
# Set to zero to use above saved model
TRAIN_EPOCHS = 10
# If you want to save the model at every epoch in a subfolder set to 'True'
SAVE_EPOCHS = False
# If you just want to save the final output in current folder, set to 'True'
SAVE_LAST = False
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

TRAIN = False
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

def getData(stockName):
    stock = Ticker(stockName)
    hist = stock.history(period="max")

    print(hist)

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print('[INFO] GPU is detected.')
    # print('[INFO] GPU is detected.', flush = True)
else:
    print('[INFO] GPU not detected.')
    # print('[INFO] GPU not detected.', flush = True)
print('[INFO] Done importing packages.')
# print('[INFO] Done importing packages.', flush = True)

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

        self.model.add(layers.Conv1D(16, 3, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        # For MaxPooling2D, default strides is equal to pool_size.  Batch and layers are assumed to match whatever comes in.
        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Conv1D(32, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))
        # # In our example, we are now at 10 x 10 x 16.
        self.model.add(layers.Conv1D(64, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        self.model.add(layers.Conv1D(128, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

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
        # print(summaryStr, file=fout)

print("[INFO] Loading Traning and Test Datasets.")
print("[INFO] Loading Traning and Test Datasets.")

getData("TSLA")

#get the boston housing training set
#test_split determiones how much of the data set to be test, and seed is a random number to randomize
((trainX, trainY), (testX, testY)) = datasets.boston_housing.load_data(test_split=0.2, seed=113)
# Convert from integers 0-255 to decimals 0-1.
# trainX = trainX.astype("float") / 255.0
# testX = testX.astype("float") / 255.0

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
# normalizer = preprocessing2.Normalization()
trainX = np.transpose(trainX)
pt = preprocessing.PowerTransformer()
trainX = pt.fit_transform(trainX)
# trainX = preprocessing.scale(trainX)
trainX = np.transpose(trainX)
# for row in trainX:
#     max = max(row)
#     min = min(row)
#     for item in row

# normalizer.adapt(trainX)

# normalizer = preprocessing2.Normalization()
testX = np.transpose(testX)
# testX = preprocessing.scale(testX)
testX = pt.fit_transform(testX)
testX = np.transpose(testX)
# normalizer.adapt(testX)

# trainX = np.transpose(trainX)
# for row in trainX:
#     print(max(row), min(row))
# trainX = np.transpose(trainX)

# print("convert to int")
# print(type(trainY))
# trainY = [int(x) for x in trainY]
# testY = [int(x) for x in testY]
# trainY = trainY.astype(int)
# testY = testY.astype(int)

# d = preprocessing.KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')
# trainY.reshape(-1, 1)
# print(trainY)
# d.fit(trainY)
# testY.reshape(-1, 1)
# d.fit(testY)

# Convert labels from integers to vectors.

# lb = preprocessing.LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# testY = lb.fit_transform(testY)

# targets = range(1,50)
# preprocessing.label_binarize(trainY, classes=targets)
# preprocessing.label_binarize(testY, classes=targets)

#trying to use conv 1d
#number of rows, columns/row, 1
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
print(trainX.shape)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
print(testX.shape)

if TRAIN:
    #this works but need to figure out why
    net=Net((13,1))
    # Notice that this will print both to console and to file.
    print(net)

    results = net.model.fit(generator(BATCH_SIZE_TRAIN, trainX, trainY), validation_data=generator(BATCH_SIZE_TEST, testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1, steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN, validation_steps=len(testX)/BATCH_SIZE_TEST)

    net.model.save("./models")

    # tf.io.write_file("newestRun.txt", results)

    #dont need the batch_size=4
    theModel = net.model.evaluate(testX, testY)

    predictions = net.model.predict(testX).flatten()

    #dont use this, i need a histogram
    # fig = plt.figure("real vs preds")
    # fig.tight_layout()
    # # plt1 = fig.add_subplot(221)
    # # plt2 = fig.add_subplot(212)
    # plt2 = fig.add_subplot()
    # # plt3 = fig.add_subplot(211)
    # # plt2.title.set_text("testing values")
    # # plt3.title.set_text("all values")
    # #put testing category on x axis and house price on y axis
    # plt2.scatter(testY, predictions, c='black', marker='*', alpha=0.5)
    # # plt2.scatter(testX, preds, c='red', marker='|', alpha=0.5, label='predicted values')
    # # plt3.scatter(dataX, dataY, c='black', marker='.', alpha=0.5)
    # # pyplot.subplots_adjust(top=1.5)
    # plt2.legend(loc='lower right')
    # plt.show()

    fig = plt.figure("preds vs real", figsize=(10, 8))
    fig.tight_layout()
    plt1 = fig.add_subplot(221)
    plt1.title.set_text("histogram of preds vs real")
    plt1.hist2d(testY, predictions, bins=100)
    plt2 = fig.add_subplot(222)
    plt2.title.set_text("best fit line of preds vs real")
    plt2.scatter(testY, predictions)
    m, b = np.polyfit(testY, predictions, 1)
    plt2.plot(testY,m*testY+b)
    plt3 = fig.add_subplot(223)
    plt3.title.set_text("training and validation loss")
    plt3.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    plt3.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    plt3.legend(loc='upper right')
    plt4 = fig.add_subplot(224)
    plt4.title.set_text("training and validation mse")
    plt4.plot(np.arange(0, TRAIN_EPOCHS), results.history['mse'],color="green", label="real")
    plt4.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_mse'], color="red", label="preds")
    plt4.legend(loc='upper right')
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

    theModel = oldModel.evaluate(testX, testY)
    predictions = oldModel.predict(testX).flatten()

    fig = plt.figure("preds vs real", figsize=(10, 4))
    fig.tight_layout()
    plt1 = fig.add_subplot(121)
    plt1.title.set_text("histogram of preds vs real")
    plt1.hist2d(testY, predictions, bins=100)
    plt2 = fig.add_subplot(122)
    plt2.title.set_text("best fit line of preds vs real")
    plt2.scatter(testY, predictions)
    m, b = np.polyfit(testY, predictions, 1)
    plt2.plot(testY,m*testY+b)
    # plt3 = fig.add_subplot(223)
    # plt3.title.set_text("training and validation loss")
    # plt3.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    # plt3.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    # plt3.legend(loc='upper right')
    # plt4 = fig.add_subplot(224)
    # plt4.title.set_text("training and validation mse")
    # plt4.plot(np.arange(0, TRAIN_EPOCHS), results.history['mse'],color="green", label="real")
    # plt4.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_mse'], color="red", label="preds")
    # plt4.legend(loc='upper right')
    plt.savefig("pyplots/newestPlot.png")
    plt.show()
