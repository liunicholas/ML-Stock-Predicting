print('[INFO] Importing packages.')
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

#TODO: make gpu work??
# devices = tf.config.list_physical_devices('GPU')
# if len(devices) > 0:
#     print('[INFO] GPU is detected.')
# else:
#     print('[INFO] GPU not detected.')

print('[INFO] Done importing packages.')

INDEX = "SPY"
indexSource = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

trainStart = "2020-01-01"
trainEnd = "2020-06-30"

testStart = "2020-07-01"
testEnd = "2020-12-31"

expectedTrain = 104
expectedTest = 107

TRAIN_EPOCHS = 50
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

LOAD_DATASET = False
TRAIN = True
TEST = True

daysBefore = 20

graphPath = "./info/pyplots/newestPlot.png"
dataPath = "./info/datasets/allSpy.npy"
checkpointPath = "./info/checkpoints"

customCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPath,
    # save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

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

def getTickers():
    table=pd.read_html(f'{indexSource}')
    df = table[0]
    dfNP = df.to_numpy()
    dfNP = np.transpose(dfNP)

    return dfNP[0]

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
    #uses daysBefore previous days
    for x in range(len(high)-daysBefore):
        X.append(high[x:x+daysBefore])

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
    #target is daysBefore+1
    for y in range(len(high)-daysBefore):
        Y.append(high[y+daysBefore])

    Y = np.array(Y)
    Y = Y.flatten()

    return Y

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

        self.model.add(layers.Conv1D(16, 3, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        # For MaxPooling2D, default strides is equal to pool_size.  Batch and layers are assumed to match whatever comes in.
        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Conv1D(64, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        self.model.add(layers.Conv1D(128, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        self.model.add(layers.Conv1D(256, 3, activation = 'relu'))
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
        self.model.add(layers.Dense(20, activation = 'relu'))
        self.model.add(layers.Dense(1))

        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.Adam(lr=0.0001)
        #absolute for regression, squared for classification

        #Absolute for few outliers
        #squared to aggresively diminish outliers
        self.loss = losses.MeanSquaredError()
        #metrics=['accuracy']
        #metrics=['mse']
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)

if LOAD_DATASET:
    print("[INFO] Loading Traning and Test Datasets.")

    #PLAN: make each row a set of stocks in an index
    INDEX_STOCKS = getTickers()
    # ["AAPL", "MSFT", "AMZN", "FB", "GOOGL", "GOOG", "TSLA", "BRK.B", "JPM", "JNJ"]

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
        print(f"train stock shape: {trainXstock.shape}")
        print(f"test stock shape: {testXstock.shape}")

        if trainXstock.shape[0] != 0:
            if trainXstock.shape[1] == expectedTrain and testXstock.shape[1] == expectedTest:
                trainXstock = trainXstock.reshape((1,20,-1))
                testXstock = testXstock.reshape((1,20,-1))
                # trainXstock = np.swapaxes(trainXstock,0,1)
                # testXstock = np.swapaxes(testXstock,0,1)
                stockHistsTrainX.append(trainXstock)
                stockHistsTestX.append(testXstock)

    print(f"[INFO] Rehaping Dataset.")
    print(f"train stock shape after reshape: {stockHistsTrainX[0].shape}")
    print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")
    # print(stockHistsTrainX[0])

    #must figure out how to stack these
    stockHistsTrainX = np.vstack(stockHistsTrainX)
    stockHistsTestX = np.vstack(stockHistsTestX)
    trainX = np.transpose(stockHistsTrainX)
    testX = np.transpose(stockHistsTestX)

    print(f"total shape before change train: {trainX.shape}")
    print(f"total shape before change test: {testX.shape}")
    trainX = np.swapaxes(trainX,1,2)
    testX = np.swapaxes(testX,1,2)
    print(f"total shape after change train: {trainX.shape}")
    print(f"total shape after change test: {testX.shape}")
    # print(trainX)

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
    print(f"target shape train: {trainY.shape}")
    print(f"target shape test: {testY.shape}")
    # print(trainY)
    # print(f"stock shape: {trainY.shape}")

    with open(f'{dataPath}', 'wb') as f:
        np.save(f, trainX)
        np.save(f, trainY)
        np.save(f, testX)
        np.save(f, testY)
else:
    with open(f'{dataPath}', 'rb') as f:
        trainX = np.load(f)
        trainY = np.load(f)
        testX = np.load(f)
        testY =  np.load(f)

if TRAIN:
    assert trainX.shape[1] == testX.shape[1]
    numStocks = trainX.shape[1]
    net=Net((numStocks, daysBefore))
    # print(net)

    results = net.model.fit(generator(BATCH_SIZE_TRAIN, trainX, trainY),
        validation_data=generator(BATCH_SIZE_TEST, testX, testY),
        shuffle = True,
        epochs = TRAIN_EPOCHS,
        batch_size = BATCH_SIZE_TRAIN,
        validation_batch_size = BATCH_SIZE_TEST,
        verbose = 1,
        steps_per_epoch=len(trainX)/BATCH_SIZE_TRAIN,
        validation_steps=len(testX)/BATCH_SIZE_TEST,
        callbacks=[customCallback])

    # net.model.load_weights(checkpointPath)
    # net.model.save("./models")

    # bestModel = tf.keras.models.load_model(checkpointPath)
    #
    # print(f"[INFO] Making Predictions.")
    # predictions = bestModel.predict(testX)
    # # print(predictions)
    #
    # fig = plt.figure("preds vs real high price", figsize=(10, 8))
    # fig.tight_layout()
    # plt1 = fig.add_subplot(211)
    # plt1.title.set_text("training and validation loss")
    # plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    # plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    # plt1.legend(loc='upper right')
    # plt2 = fig.add_subplot(212)
    # histTestIndex = histTestIndex.iloc[20:]
    # plt2.plot(histTestIndex.index, testY, color="blue", label="real 2020")
    # plt2.plot(histTestIndex.index, predictions, color="red", label="preds 2020")
    # plt2.legend(loc='upper right')
    # plt.savefig("pyplots/newestPlot.png")
    # plt.show()

if TEST:
    histTestIndex = getData(f"{INDEX}", testStart, testEnd)
    bestModel = tf.keras.models.load_model(checkpointPath)

    print(f"[INFO] Making Predictions.")
    predictions = bestModel.predict(testX)
    # print(predictions)

    fig = plt.figure("preds vs real high price", figsize=(10, 8))
    fig.tight_layout()
    plt1 = fig.add_subplot(211)
    plt1.title.set_text("training and validation loss")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
    plt1.legend(loc='upper right')
    plt2 = fig.add_subplot(212)
    histTestIndex = histTestIndex.iloc[daysBefore:]
    plt2.plot(histTestIndex.index, testY, color="blue", label="real")
    plt2.plot(histTestIndex.index, predictions, color="red", label="preds")
    plt2.legend(loc='upper right')
    plt.savefig(graphPath)
    plt.show()
