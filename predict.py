print('[INFO] Importing packages.')
# Python                 3.8.1
# Keras-Preprocessing    1.1.2
# matplotlib             3.3.3
# multitasking           0.0.9
# numpy                  1.18.5
# pandas                 1.2.2
# scikit-learn           0.24.1
# scipy                  1.6.0
# tensorflow             2.3.1
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
from datetime import *

#TODO: make gpu work??
# devices = tf.config.list_physical_devices('GPU')
# if len(devices) > 0:
#     print('[INFO] GPU is detected.')
# else:
#     print('[INFO] GPU not detected.')

#STUFF TO DO
#normalization
#predict future
#check for the extra dates
#gpu

print('[INFO] Done importing packages.')

#the 9 federally recognized holidays
holidays2021 = ["2021-01-01", "2021-01-18", "2021-02-15",
    "2021-04-02", "2021-05-31", "2021-07-05",
    "2021-09-06", "2021-11-25", "2021-12-25"]

INDEX = "SPY"
indexSource = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#to not have to get all 500+ stocks
shortList = ["AAPL", "MSFT", "AMZN", "FB", "GOOGL", "GOOG", "TSLA", "BRK.B", "JPM", "JNJ"]

trainStart = "2019-03-12"
trainEnd = "2020-3-11"

testStart = "2020-03-11"
testEnd = "2021-3-11"

expectedTrain = 232
expectedTest = 232

TRAIN_EPOCHS = 10
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

USE_ALL_STOCKS = True
LOAD_DATASET = False

TRAIN = False
TEST = False

PREDICT = True
NEW_MODEL = False
predictDate = "2021-03-15"
daysBefore = 20                 #total days in period for prediction
daysAhead = 1                   #1 for day immediately after

graphPath = "./info/pyplots/newestPlot.png"
dataPath = "./info/datasets/allSpy.npy"
checkpointPath = "./info/checkpoints"
stocksIncludedPath = "./info/datasets/stocksIncluded.txt"
previousSavePath = "./info/savedModels/20_1_480/"
previousSavestocksIncludedPath = "./info/savedModels/20_1_480/stocksIncluded.txt"

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
    for x in range(len(high)-daysBefore-daysAhead+1):
        X.append(high[x:x+daysBefore])

    X = np.array(X)

    return X

def getXpredict(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    high = histNP[1]

    return np.array([high])

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
    for y in range(len(high)-daysBefore-daysAhead+1):
        Y.append(high[y+daysBefore+daysAhead-1])

    Y = np.array(Y)
    Y = Y.flatten()

    return Y

def getDateInPast(initial, days):
    #inspiration from https://stackoverflow.com/a/12691993
    daysSubtract = days
    currentDate = datetime.strptime(initial, "%Y-%m-%d")
    while daysSubtract > 0:
        currentDate -= timedelta(days=1)
        #means it's a weekend and won't count
        if currentDate.weekday() >= 5:
            continue
        if currentDate.strftime("%Y-%m-%d") in holidays2021:
            continue
        daysSubtract -= 1

    return currentDate

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

def main():
    if LOAD_DATASET:
        print("[INFO] Loading Traning and Test Datasets.")

        if USE_ALL_STOCKS:
            INDEX_STOCKS = getTickers()
        else:
            INDEX_STOCKS = shortList

        f = open(f"{stocksIncludedPath}", 'w')

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
                if trainXstock.shape[1] != expectedTrain or testXstock.shape[1] != expectedTest:
                    print("possible error: did not set expectedTrain and expectedTest")
                    if testXstock.shape[0] != expectedTest:
                        while testXstock.shape[0] < expectedTest:
                            print(testXstock)
                            testXstock = np.vstack([testXstock[0],testXstock])
                            print(testXstock)
                        while testXstock.shape[0] > dexpectedTest:
                            print(testXstock)
                            testXstock = np.delete(testXstock,expectedTest)
                            print(testXstock)

                    if trainXstock.shape[0] != expectedTrain:
                        while trainXstock.shape[0] < expectedTrain:
                            print(trainXstock)
                            testXstock = np.vstack([trainXstock[0],trainXstock])
                            print(testXstock)
                        while trainXstock.shape[0] > expectedTrain:
                            print(trainXstock)
                            trainXstock = np.delete(trainXstock,expectedTrain)
                            print(trainXstock)


                f.write(f" {stock} ")
                trainXstock = trainXstock.reshape((1,daysBefore,-1))
                testXstock = testXstock.reshape((1,daysBefore,-1))
                # trainXstock = np.swapaxes(trainXstock,0,1)
                # testXstock = np.swapaxes(testXstock,0,1)
                stockHistsTrainX.append(trainXstock)
                stockHistsTestX.append(testXstock)

        f.close()

        print(f"[INFO] Rehaping Dataset.")
        print(f"train stock shape after reshape: {stockHistsTrainX[0].shape}")
        print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")

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

        #index target prices
        print("[INFO] Loading Index Data.")
        histTrainIndex = getData(f"{INDEX}", trainStart, trainEnd)
        histTestIndex = getData(f"{INDEX}", testStart, testEnd)
        trainY = getY(histTrainIndex)
        testY = getY(histTestIndex)
        print(f"target shape train: {trainY.shape}")
        print(f"target shape test: {testY.shape}")

        with open(f'{dataPath}', 'wb') as f:
            np.save(f, trainX)
            np.save(f, trainY)
            np.save(f, testX)
            np.save(f, testY)

    if TRAIN:
        if not LOAD_DATASET:
            with open(f'{dataPath}', 'rb') as f:
                trainX = np.load(f)
                trainY = np.load(f)
                testX = np.load(f)
                testY =  np.load(f)

        assert trainX.shape[1] == testX.shape[1]
        numStocks = trainX.shape[1]
        net=Net((numStocks, daysBefore))

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

    if TEST:
        if not LOAD_DATASET and not TRAIN:
            with open(f'{dataPath}', 'rb') as f:
                trainX = np.load(f)
                trainY = np.load(f)
                testX = np.load(f)
                testY =  np.load(f)
        histTestIndex = getData(f"{INDEX}", testStart, testEnd)
        histTestIndex = histTestIndex.iloc[daysBefore+daysAhead-1:]

        bestModel = tf.keras.models.load_model(checkpointPath)

        print(f"[INFO] Making Predictions.")
        predictions = bestModel.predict(testX)

        counter = 0
        for date in histTestIndex.index:
            print(f"{date}: {predictions[counter]}")
            counter+=1

        if TRAIN:
            fig = plt.figure("preds vs real high price", figsize=(10, 8))
            fig.tight_layout()
            plt1 = fig.add_subplot(211)
            plt1.title.set_text("training and validation loss")
            plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="real")
            plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="preds")
            plt1.legend(loc='upper right')
            plt2 = fig.add_subplot(212)
            plt2.plot(histTestIndex.index, testY, color="blue", label="real")
            plt2.plot(histTestIndex.index, predictions, color="red", label="preds")
            plt2.legend(loc='upper left')
            plt.savefig(graphPath)
            plt.show()
        else:
            fig = plt.figure("preds vs real high price", figsize=(10, 4))
            fig.tight_layout()
            plt2 = fig.add_subplot(111)
            plt2.plot(histTestIndex.index, testY, color="blue", label="real")
            plt2.plot(histTestIndex.index, predictions, color="red", label="preds")
            plt2.legend(loc='upper left')
            plt.savefig(graphPath)
            plt.show()

    if PREDICT:
        predictStart = getDateInPast(predictDate,daysAhead+daysBefore-1)
        print(f"predict start date: {predictStart}")
        predictEnd = getDateInPast(predictDate, daysAhead-1)
        print(f"predict end date: {predictEnd}")

        if NEW_MODEL:
            f = open(f"{stocksIncludedPath}", 'r')
            stocksIncluded = f.read()
            f.close()
        else:
            f = open(f"{previousSavestocksIncludedPath}", 'r')
            stocksIncluded = f.read()
            f.close()

        print("[INFO] Loading Prediction Datasets.")

        if USE_ALL_STOCKS:
            INDEX_STOCKS = getTickers()
        else:
            INDEX_STOCKS = shortList

        stockHistsTestX = []
        for stock in INDEX_STOCKS:
            if " " + stock + " " in stocksIncluded:
                print(f"[INFO] Loading Testset For {stock}.")
                test = getData(f"{stock}", predictStart, predictEnd)
                testXstock = np.transpose(getXpredict(test))
                print(f"test stock shape: {testXstock.shape}")
                if testXstock.shape[0] != daysBefore:
                    while testXstock.shape[0] < daysBefore:
                        print(testXstock)
                        testXstock = np.vstack([testXstock[0],testXstock])
                        print(testXstock)
                    while testXstock.shape[0] > daysBefore:
                        print(testXstock)
                        testXstock = np.delete(testXstock,daysBefore)
                        print(testXstock)
                testXstock = testXstock.reshape((1,daysBefore,-1))
                stockHistsTestX.append(testXstock)

        print(f"[INFO] Rehaping Testset.")
        print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")

        stockHistsTestX = np.vstack(stockHistsTestX)
        testX = np.transpose(stockHistsTestX)

        print(f"total shape before change test: {testX.shape}")
        testX = np.swapaxes(testX,1,2)
        print(f"total shape after change test: {testX.shape}")

        if NEW_MODEL:
            bestModel = tf.keras.models.load_model(checkpointPath)
        else:
            bestModel = tf.keras.models.load_model(previousSavePath)

        print(f"[INFO] Making Prediction.")
        prediction = bestModel.predict(testX)

        print(f"\nprediction for {predictDate}: {prediction[0][0]}\n")

main()
