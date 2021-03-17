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
import os

#TODO: make gpu work??
# devices = tf.config.list_physical_devices('GPU')
# if len(devices) > 0:
#     print('[INFO] GPU is detected.')
# else:
#     print('[INFO] GPU not detected.')

#STUFF TO DO
#natural log of dataset
#gpu get cuda 11

print('[INFO] Done importing packages.')

#the 9 federally recognized holidays
holidays2021 = ["2021-01-01", "2021-01-18", "2021-02-15",
    "2021-04-02", "2021-05-31", "2021-07-05",
    "2021-09-06", "2021-11-25", "2021-12-25"]

#focusing on SPDR S&P 500 ETF
INDEX = "SPY"
indexSource = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
shortList = ["AAPL", "MSFT", "AMZN", "FB", "GOOGL",
    "GOOG", "TSLA", "BRK.B", "JPM", "JNJ"]

#dates for training and testing range
trainStart = "2019-03-12"
trainEnd = "2020-9-11"

testStart = "2020-09-11"
testEnd = "2021-3-11"

LOAD_DATASET = True           #set to false when testing architecture
USE_ALL_STOCKS = True         #set to false for just testing
OHLC = 1                      #open = 0, high = 1, low = 2, close = 3

daysBefore = 21                #total days in period for prediction
daysAhead = 7                  #total days predicting in future
expectedTrain = 353            #find with test run
expectedTest = 97             #find with test run

QUICK_RUN = False              #for just testing code 

TRAIN = True
TRAIN_EPOCHS = 10
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

TEST = True
NEW_MODEL = True              #tests on a new model

PREDICT_ON_DATE = False        #set to true to predict day
OVERRIDE = True                #overrides load, train, test, and new_model

#vars for predicting
predictDate = "2021-03-25"
savedModelName = "21_7_7daysFuture"

graphPath = "./info/pyplots/newestPlot.png"                  #save mpl graph
dataPath = "./info/datasets/allSpy.npy"                      #save npy arrays
checkpointPath = "./info/checkpoints"                        #save models
stocksIncludedPath = "./info/datasets/stocksIncluded.txt"    #save list of stocks used
savedModelsPath = "./savedModels"                            #save best model
previousSavePath = f"{savedModelsPath}/{savedModelName}/"    #location of desired model for predicting
previousStocksIncludedPath = f"{savedModelsPath}/{savedModelName}/stocksIncluded.txt"

#overides load, train, test, when predicting
def setModes():
    #only makes new global variables if needed
    if PREDICT_ON_DATE and OVERRIDE or QUICK_RUN:
        global LOAD_DATASET
        global TRAIN
        global TEST
        global NEW_MODEL

    #overrides and only predicts price on date
    if PREDICT_ON_DATE and OVERRIDE:
        LOAD_DATASET = False
        TRAIN = False
        TEST = False
        NEW_MODEL = False           #set this to true when desperate

    #use for testing features (does not test predict)
    if QUICK_RUN:
        LOAD_DATASET = True
        global USE_ALL_STOCKS
        USE_ALL_STOCKS = False
        TRAIN = True
        global TRAIN_EPOCHS
        TRAIN_EPOCHS = 2
        TEST = True
        NEW_MODEL = True
#to only save the best model after each epoch
def setCustomCallback():
    global customCallback
    customCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        # save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

#get the tickers of stocks that SPY tracks
def getTickers():
    table=pd.read_html(f'{indexSource}')
    df = table[0]
    dfNP = df.to_numpy()
    dfNP = np.transpose(dfNP)

    return dfNP[0]
#get pandas dataframe of a stock
def getData(stockName, startDate, endDate):
    stock = Ticker(stockName)
    hist = stock.history(start=startDate, end=endDate)

    return hist

#get a formatted dataset of OHLC of stock as numpy array
def getXnumpy(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    high = histNP[OHLC]

    X = []
    #uses daysBefore previous days
    for x in range(len(high)-daysBefore-daysAhead+1):
        X.append(high[x:x+daysBefore])

    X = np.array(X)

    return X
#get a formatted dataset of OHLC of stock as numpy array
def getXnumpyPredict(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    high = histNP[OHLC]

    return np.array([high])
#get a formatted dataset of OHLC of index as numpy array
def getYnumpy(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    high = histNP[OHLC]

    Y = []
    #target is daysBefore+1
    for y in range(len(high)-daysBefore-daysAhead+1):
        Y.append(high[y+daysBefore+daysAhead-1])

    Y = np.array(Y)
    Y = Y.flatten()

    return Y

#locate and replace NaN in numpy array
def removeNaN(stockArray):
    for rowIndex in range(stockArray.shape[0]):
        sum = 0
        counter = 0
        for val in stockArray[rowIndex]:
            if not np.isnan(val):
                sum += val
                counter += 1
        avg = sum/counter

        for i in range(stockArray[rowIndex].shape[0]):
            if np.isnan(stockArray[rowIndex][i]):
                print("NaN Found")
                stockArray[rowIndex][i] = avg

    return stockArray
#adds or removes data as necessary to fit expected train and test values
def fixData(stockArray, expectedVal):
    stockArray = np.transpose(stockArray)
    while stockArray.shape[0] < expectedVal:
        stockArray = np.vstack([stockArray[0],stockArray])
    while stockArray.shape[0] > expectedVal:
        stockArray = np.delete(stockArray,0,0)
    stockArray = np.transpose(stockArray)

    return stockArray
#vstack, transpose, swaps axis 1 and 2
def stackTransposeSwap(stocksList):
    stocksCombined = np.vstack(stocksList)
    xSet = np.transpose(stocksCombined)
    xSet = np.swapaxes(xSet,1,2)

    return xSet

#get number of stocks used in dataset
def getNumStocks(testX, trainX):
    assert trainX.shape[1] == testX.shape[1]
    numStocks = trainX.shape[1]

    return numStocks
#won't load all data at once while training and testing
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

#save numpy arrays to npy file
def saveDataSet(dataPath, trainX, trainY, testX, testY):
    with open(dataPath, 'wb') as f:
        np.save(f, trainX)
        np.save(f, trainY)
        np.save(f, testX)
        np.save(f, testY)
#loads numpy arrays from npy file
def loadDataSet(dataPath):
    with open(dataPath, 'rb') as f:
        trainX = np.load(f)
        trainY = np.load(f)
        testX = np.load(f)
        testY =  np.load(f)

    return trainX, trainY, testX, testY

#read contents of text file
def readFile(filePath):
    f = open(filePath, 'r')
    contents = f.read()
    f.close()

    return contents
#write contents to text file
def writeFile(filePath, contents):
    f = open(filePath, 'w')
    f.write(contents)
    f.close()

#displays prediction with date followed by prediction
def displayPredictionsAsText(histTestIndex, predictions):
    counter = 0
    for date in histTestIndex.index:
        print(f"{date}: {predictions[counter]}")
        counter+=1

#ask user if they want to save the model
def askUserSaveModel():
    while True:
        keep = input("save this model to folder? (y/n)")
        if keep != "y" and keep != "n":
            print("error")
            continue
        break

    return keep
#ask user for version name
def getVersionName():
    while True:
        version = input("version name: ")
        while True:
            confirm = input("confirm? (y/n)")
            if confirm != "y" and confirm != "n":
                print("error")
                continue
            break
        if confirm == "y":
            break

    return version

#makes new folder for saved model
def makeNewFolder(version):
    print("[INFO] Making New Model Folder.")
    newFolderPath = f"{savedModelsPath}/{daysBefore}_{daysAhead}_{version}"
    os.mkdir(newFolderPath)

    return newFolderPath
#saves pyplot to folder for later analysis
def savePyPlot(newFolderPath, version, histTestIndex, testY, predictions):
    print("[INFO] Saving Pyplot.")
    fig = getJustPriceGraph(histTestIndex, testY, predictions)
    plt.savefig(f"{newFolderPath}/{daysBefore}_{daysAhead}_{version}.png")
#saves text file of included stocks to folder
def saveIncludedStocks(newFolderPath):
    print("[INFO] Saving Included Stocks Text File.")
    stocksIncluded = readFile(stocksIncludedPath)
    writeFile(f"{newFolderPath}/stocksIncluded.txt", stocksIncluded)
#saves best model to folder
def saveModel(newFolderPath, bestModel):
    print("[INFO] Saving Model.")
    bestModel.save(newFolderPath)
#saves all info to text file
def saveParameters(newFolderPath, version, numStocks):
    print("[INFO] Saving Parameters.")
    f = open(f"{newFolderPath}/{daysBefore}_{daysAhead}_{version} info.txt", 'w')
    f.write(f"version name: {daysBefore}_{daysAhead}_{version}\n")
    f.write(f"training dates: {trainStart} to {trainEnd}\n")
    f.write(f"testing dates: {testStart} to {testEnd}\n")
    f.write(f"days before: {daysBefore} days ahead: {daysAhead} \n")
    f.write(f"number of stocks included: {numStocks}\n")
    f.close()

#get loss and price graph
def getLossAndPriceGraph(results, histTestIndex, testY, predictions):
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

    return fig
#get real vs preds price graph
def getJustPriceGraph(histTestIndex, testY, predictions):
    fig = plt.figure("preds vs real high price", figsize=(10, 4))
    fig.tight_layout()
    plt2 = fig.add_subplot(111)
    plt2.plot(histTestIndex.index, testY, color="blue", label="real")
    plt2.plot(histTestIndex.index, predictions, color="red", label="preds")
    plt2.legend(loc='upper left')

    return fig

#get days before and ahead from saved model name
def parseDaysBeforeAndAhead():
    index1 = savedModelName.find("_")
    daysBefore = int(savedModelName[:index1])
    print(daysBefore)
    savedModelName2 = savedModelName[index1+1:]
    index2 = savedModelName2.find("_")
    daysAhead = int(savedModelName2[:index2])
    print(daysAhead)

    return daysBefore, daysAhead
#get x days in past excluding holidays and weekends
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
class CNN():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        # For Conv2D, you give it: Outgoing Layers, Frame size.  Everything else needs a keyword.
        # Popular keyword choices: strides (default is strides=1), padding (="valid" means 0, ="same" means whatever gives same output width/height as input).  Not sure yet what to do if you want some other padding.
        # Activation function is built right into the Conv2D function as a keyword argument.

        self.model.add(layers.Conv1D(16, 3, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Conv1D(64, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        self.model.add(layers.Conv1D(128, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        self.model.add(layers.Conv1D(256, 3, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))

        # self.model.add(layers.MaxPooling2D(pool_size = 2))

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
    setModes()
    setCustomCallback()

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
            train = removeNaN(getXnumpy(train))
            test = removeNaN(getXnumpy(test))
            trainXstock = np.transpose(train)
            testXstock = np.transpose(test)
            print(f"train stock shape: {trainXstock.shape}")
            print(f"test stock shape: {testXstock.shape}")

            if trainXstock.shape[0] != 0 and testXstock.shape[0] != 0:
                if trainXstock.shape[1] != expectedTrain or testXstock.shape[1] != expectedTest:
                    print("possible error: did not set expectedTrain and expectedTest")

                    trainXstock = fixData(trainXstock, expectedTrain)
                    testXstock = fixData(trainXstock, expectedTrain)

                    print("sketch fix, revised array shape below")
                    print(f"train stock shape: {trainXstock.shape}")
                    print(f"test stock shape: {testXstock.shape}")

                if trainXstock.shape[1] == expectedTrain and testXstock.shape[1] == expectedTest:
                    f.write(f" {stock} ")
                    trainXstock = trainXstock.reshape((1,daysBefore,-1))
                    testXstock = testXstock.reshape((1,daysBefore,-1))
                    stockHistsTrainX.append(trainXstock)
                    stockHistsTestX.append(testXstock)
        f.close()

        print(f"[INFO] Rehaping Dataset.")
        print(f"train stock shape after reshape: {stockHistsTrainX[0].shape}")
        print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")

        trainX = stackTransposeSwap(stockHistsTrainX)
        testX = stackTransposeSwap(stockHistsTestX)

        print(f"total shape after change train: {trainX.shape}")
        print(f"total shape after change test: {testX.shape}")

        #index target prices
        print("[INFO] Loading Index Data.")
        histTrainIndex = getData(f"{INDEX}", trainStart, trainEnd)
        histTestIndex = getData(f"{INDEX}", testStart, testEnd)
        trainY = getYnumpy(histTrainIndex)
        testY = getYnumpy(histTestIndex)
        print(f"target shape train: {trainY.shape}")
        print(f"target shape test: {testY.shape}")

        saveDataSet(dataPath, trainX, trainY, testX, testY)

    if TRAIN:
        #load dataset if didn't load dataset in current execution
        if not LOAD_DATASET:
            trainX, trainY, testX, testY = loadDataSet(dataPath)

        numStocks = getNumStocks(testX, trainX)
        cnn = CNN((numStocks, daysBefore))

        results = cnn.model.fit(generator(BATCH_SIZE_TRAIN, trainX, trainY),
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
        #load dataset if didn't load dataset in current execution
        if not LOAD_DATASET and not TRAIN:
            trainX, trainY, testX, testY = loadDataSet(dataPath)

        histTestIndex = getData(f"{INDEX}", testStart, testEnd)
        histTestIndex = histTestIndex.iloc[daysBefore+daysAhead-1:]

        if NEW_MODEL:
            bestModel = tf.keras.models.load_model(checkpointPath)
        else:
            bestModel = tf.keras.models.load_model(previousSavePath)

        print(f"[INFO] Making Predictions.")
        predictions = bestModel.predict(testX)
        displayPredictionsAsText(histTestIndex, predictions)

        if TRAIN:
            fig = getLossAndPriceGraph(results, histTestIndex, testY, predictions)
            plt.savefig(graphPath)
            plt.show()
        else:
            fig = getJustPriceGraph(histTestIndex, testY, predictions)
            plt.savefig(graphPath)
            plt.show()

        #ask to save model if new model
        if NEW_MODEL:
            keep = askUserSaveModel()
            if keep == "y":
                version = getVersionName()

                newFolderPath = makeNewFolder(version)
                savePyPlot(newFolderPath, version, histTestIndex, testY, predictions)
                saveIncludedStocks(newFolderPath)
                saveModel(newFolderPath, bestModel)

                numStocks = getNumStocks(testX, trainX)
                saveParameters(newFolderPath, version, numStocks)

    if PREDICT_ON_DATE:
        before, ahead = parseDaysBeforeAndAhead()

        predictStart = getDateInPast(predictDate,ahead+before-1)
        print(f"predict start date: {predictStart}")
        predictEnd = getDateInPast(predictDate, ahead-1)
        print(f"predict end date: {predictEnd}")

        if NEW_MODEL:
            stocksIncluded = readFile(stocksIncludedPath)
        else:
            stocksIncluded = readFile(previousStocksIncludedPath)

        print("[INFO] Loading Prediction Datasets.")

        INDEX_STOCKS = getTickers()

        stockHistsTestX = []
        for stock in INDEX_STOCKS:
            if " " + stock + " " in stocksIncluded:
                print(f"[INFO] Loading Testset For {stock}.")
                test = getData(f"{stock}", predictStart, predictEnd)
                testXstock = np.transpose(getXnumpyPredict(test))
                print(f"test stock shape: {testXstock.shape}")

                #remove NaN from dataset
                sum = 0
                counter = 0
                for val in testXstock:
                    if not np.isnan(val[0]):
                        sum += val[0]
                        counter += 1
                avg = sum/counter

                for index in range(testXstock.shape[0]):
                    if np.isnan(testXstock[index][0]):
                        print("NaN Found")
                        testXstock[index][0] = avg

                if testXstock.shape[0] != before:
                    while testXstock.shape[0] < before:
                        print(testXstock)
                        testXstock = np.vstack([testXstock[0],testXstock])
                        print(testXstock)
                    while testXstock.shape[0] > before:
                        print(testXstock)
                        testXstock = np.delete(testXstock,0)
                        print(testXstock)

                testXstock = testXstock.reshape((1,before,-1))
                stockHistsTestX.append(testXstock)

        print(f"[INFO] Rehaping Testset.")
        print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")

        testX = stackTransposeSwap(stockHistsTestX)
        print(f"total shape after change test: {testX.shape}")

        if NEW_MODEL:
            bestModel = tf.keras.models.load_model(checkpointPath)
        else:
            bestModel = tf.keras.models.load_model(previousSavePath)

        print(f"[INFO] Making Prediction.")
        prediction = bestModel.predict(testX)

        print(f"\nprediction for {predictDate}: {prediction[0][0]}\n")

main()
