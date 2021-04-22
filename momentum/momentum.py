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

print('[INFO] Done importing packages.')

#work on getXnumpy and getYnumpy and play with intervals
#work on architecture [ask dr j]
#change parse before and ahead to intervals [written but not tested]
#fix save Parameters [written but not tested]
#make graphs [make graph of momentum vs buy or sell]
#fix save user model

holidays2021 = ["2021-01-01", "2021-01-18", "2021-02-15",
    "2021-04-02", "2021-05-31", "2021-07-05",
    "2021-09-06", "2021-11-25", "2021-12-25"]

STOCK = "TSLA"

#set QUICK_RUN to true for quick testing
#set PREDICT_ON_DATE to true and OVERRIDE to true for just predicting a date

#dates for training and testing range
trainStart = "2015-01-01"
trainEnd = "2018-12-31"

testStart = "2019-01-01"
testEnd = "2021-3-11"

LOAD_DATASET = True          #set to false when testing architecture
OHLC = 1                      #open = 0, high = 1, low = 2, close = 3

#intervals are total days not days before
#add intervals and subtract 2 to get start values for data needed
intervalMomentum = 10          #interval to find momentum
intervalPeriod = 5            #interval to group together

daysAhead = 1                  #predict days ahead

# expectedTrain = 267            #find with test run
# expectedTest = 224             #find with test run

QUICK_RUN = False              #for just testing code

TRAIN = True
TRAIN_EPOCHS = 5
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16

TEST = True
NEW_MODEL = True              #tests on a new model

PREDICT_ON_DATE = False        #set to true to predict day
OVERRIDE = True                #overrides load, train, test, and new_model

#vars for predicting
predictDate = "2021-03-17"
savedModelName = ""

graphPath = "./info/pyplots/newestPlot.png"                  #save mpl graph
dataPath = "./info/datasets/thisStock.npy"                    #save npy arrays
checkpointPath = "./info/checkpoints"                        #save models

savedModelsPath = "./savedModels"                            #save best model
previousSavePath = f"{savedModelsPath}/{savedModelName}/"    #location of desired model for predicting

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
        monitor='val_accuracy',
        mode='max',
        # monitor='val_min',
        # mode='min',
        save_best_only=True)

#get pandas dataframe of a stock
def getData(stockName, startDate, endDate):
    stock = Ticker(stockName)
    hist = stock.history(start=startDate, end=endDate)

    return hist

#get a formatted dataset of OHLC of stock x as numpy array
def getXnumpy(hist):
    print("[INFO] getting x hist data")
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    OHLCcolumn = histNP[OHLC]
    OHLCcolumn = removeNaN(OHLCcolumn)

    # totalVals = OHLCcolumn.shape[0]-interval+1

    #get all the momentums over the data range
    #uses percentage change method
    #try subtracting method later
    momentums = []
    #intervals are total days not days before
    print("[INFO] getting momentums")
    #algorithm correct
    for i in range(intervalMomentum-1, OHLCcolumn.shape[0]):
        # momentums.append((OHLCcolumn[i]/OHLCcolumn[i-intervalMomentum+1])*100)
        momentums.append(OHLCcolumn[i]-OHLCcolumn[i-intervalMomentum+1])

    momentumGroups = []
    print("[INFO] grouping momentums")
    #algorithm correct
    for i in range(intervalPeriod-1, len(momentums)-daysAhead):
        momentumGroups.append(momentums[i-intervalPeriod+1:i+1])

    return np.array(momentumGroups)
#get a formatted dataset of OHLC of target as numpy array
def getYnumpy(hist):
    print("[INFO] getting y hist data")
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    OHLCcolumn = histNP[OHLC]
    OHLCcolumn = removeNaN(OHLCcolumn)

    startIndex = intervalMomentum+intervalPeriod-2
    # startIndex =
    binarizedList = []
    print("[INFO] classifying buy and sell")
    for i in range(startIndex, OHLCcolumn.shape[0]-daysAhead):
        #intervals are total days not days before
        # difference = OHLCcolumn[i]-OHLCcolumn[i-intervalPeriod+1]
        difference = OHLCcolumn[i+daysAhead] - OHLCcolumn[i]

        #0 for sell, 1 for buy
        #try with 0 for buy, 1 for hold, and 2 for sell later
        if difference < 0:
            binarizedList.append(0)
        else:
            binarizedList.append(1)

    # get the differences between dates and define classes

    return binarizedList
#locate and remove NaN in 1 dimensional numpy array
def removeNaN(array):
    for i in range(array.shape[0]):
        # print(array.shape[0])
        if np.isnan(array[i]):
            print("NaN Found")
            if i==0:
                array[i] = array[i+1]
                continue
            if i==array.shape[0]-1:
                array[i] = array[i-1]
                continue
            array[i] = (array[i+1] + array[i-1]) / 2

    return array

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

#gets the mpl fig for the loss
def getLossGraph(results):
    fig = plt.figure("training metrics", figsize=(10, 8))
    fig.tight_layout()
    plt1 = fig.add_subplot(211)
    plt1.title.set_text("training and validation loss")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="train")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="test")
    plt1.legend(loc='upper right')
    plt1 = fig.add_subplot(212)
    plt1.title.set_text("training and validation accuracy")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['accuracy'], color="green", label="train")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_accuracy'], color="red", label="test")
    plt1.legend(loc='upper right')

    return fig
#compare predictions with real
def evalPreds(predictions, testY):
    totalSell = 0
    correctSell = 0
    totalBuy = 0
    correctBuy = 0

    assert len(predictions) == len(testY)
    for i in range(len(predictions)):
        #add counters to find total of the acurate sell and buy indicators
        if testY[i] == 0:
            totalSell += 1
        elif testY[i] == 1:
            totalBuy += 1
        else:
            print("[WARNING] Soemthing isn't right.")

        # maxIndex = predictions[i].index(max(predictions[i]))
        maxIndex = np.argmax(predictions[i])
        if maxIndex == testY[i]:
            if testY[i] == 0:
                correctSell += 1
            elif testY[i] == 1:
                correctBuy += 1
            else:
                print("[WARNING] Soemthing isn't right.")

    print(f"Sell Accuracy: {correctSell}/{totalSell} ({round(correctSell/totalSell*100, 2)}%)")
    print(f"Buy Accuracy: {correctBuy}/{totalBuy} ({round(correctBuy/totalBuy*100, 2)}%)\n")

    return

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
    newFolderPath = f"{savedModelsPath}/{intervalMomentum}_{intervalPeriod}_{daysAhead}_{version}"
    os.mkdir(newFolderPath)

    return newFolderPath
#saves best model to folder
def saveModel(newFolderPath, bestModel):
    print("[INFO] Saving Model.")
    bestModel.save(newFolderPath)
#saves pyplot to folder for later analysis
def savePyPlot(results):
    print("[INFO] Saving Pyplot.")
    fig = getLossGraph(results)
    plt.savefig(f"{newFolderPath}/{intervalMomentum}_{intervalPeriod}_{daysAhead}_{version}.png")
#saves all info to text file
def saveParameters(newFolderPath, version):
    print("[INFO] Saving Parameters.")
    f = open(f"{newFolderPath}/{intervalMomentum}_{intervalPeriod}_{daysAhead}_{version} info.txt", 'w')
    f.write(f"version name: {intervalMomentum}_{intervalPeriod}_{daysAhead}_{version}\n")
    f.write(f"training dates: {trainStart} to {trainEnd}\n")
    f.write(f"testing dates: {testStart} to {testEnd}\n")
    f.write(f"intervalMomentum: {intervalMomentum} intervalPeriod: {intervalPeriod}")
    f.write(f"days ahead: {daysAhead} \n")
    f.close()

#get days before and ahead from saved model name
#change this to parse intervals
def parseIntervals():
    index1 = savedModelName.find("_")
    intervalMomentum = int(savedModelName[:index1])
    print(f"parsed intervalMomentum: {intervalMomentum}")
    savedModelName2 = savedModelName[index1+1:]
    index2 = savedModelName2.find("_")
    intervalPeriod = int(savedModelName2[:index2])
    print(f"parsed intervalPeriod: {intervalPeriod}")
    savedModelName3 = savedModelName[index2+1:]
    index3 = savedModelName3.find("_")
    intervalPeriod = int(savedModelName3[:index4])
    print(f"parsed daysAhead: {daysAhead}")

    return intervalMomentum, intervalPeriod, daysAhead
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

#CNN for 3D numpy array
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

        # self.model.add(layers.Conv1D(128, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        #
        # self.model.add(layers.Conv1D(256, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        #
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
        self.model.add(layers.Dense(2, activation = 'softmax'))

        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.SGD(lr=0.0001)
        #absolute for regression, squared for classification

        #Absolute for few outliers
        #squared to aggresively diminish outliers
        # self.loss = losses.MeanSquaredError()
        self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        #metrics=['accuracy']
        #metrics=['mse']
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["accuracy"])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)

#load and format dataset
def loadData():
    print("[INFO] Loading Traning and Test Datasets.")
    print(f"[INFO] Loading Dataset For {STOCK}.")

    train = getData(f"{STOCK}", trainStart, trainEnd)
    test = getData(f"{STOCK}", testStart, testEnd)
    trainX = getXnumpy(train)
    testX = getXnumpy(test)
    # if train.shape[0] != 0 and test.shape[0] != 0:
    #     train = preprocessing.normalize(train)
    #     test = preprocessing.normalize(test)
    # train = removeNaN(getXnumpy(train))
    # test = removeNaN(getXnumpy(test))
    # trainXstock = np.transpose(train)
    # print(trainXstock)
    # testXstock = np.transpose(test)

    #error with diviison by zero
    # trainXstock = pt.fit_transform(trainXstock)
    # testXstock = pt.fit_transform(testXstock)

    print(f"train stock shape: {trainX.shape}")
    print(f"test stock shape: {testX.shape}")

    # if trainXstock.shape[0] != 0 and testXstock.shape[0] != 0:
    #     if trainXstock.shape[1] != expectedTrain or testXstock.shape[1] != expectedTest:
    #         print("possible error: did not set expectedTrain and expectedTest")
    #
    #         trainXstock = fixData(trainXstock, expectedTrain)
    #         testXstock = fixData(testXstock, expectedTest)
    #
    #         print("sketch fix, revised array shape below")
    #         print(f"train stock shape: {trainXstock.shape}")
    #         print(f"test stock shape: {testXstock.shape}")
    #
    #     if trainXstock.shape[1] == expectedTrain and testXstock.shape[1] == expectedTest:
    #         f.write(f" {stock} ")
    #         trainXstock = trainXstock.reshape((1,daysBefore,-1))
    #         testXstock = testXstock.reshape((1,daysBefore,-1))
    #         stockHistsTrainX.append(trainXstock)
    #         stockHistsTestX.append(testXstock)
    # f.close()

    print(f"[INFO] Rehaping Dataset.")
    # print(f"train stock shape after reshape: {trainX.shape}")
    # print(f"test stock shape after reshape: {testX.shape}")

    # trainX = stackTransposeSwap(stockHistsTrainX)
    # testX = stackTransposeSwap(stockHistsTestX)

    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    # print(trainX.shape)
    testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
    # print(testX.shape)

    print(f"total shape after change train: {trainX.shape}")
    print(f"total shape after change test: {testX.shape}")

    #index target prices
    print("[INFO] Loading Index Data.")
    histTrainIndex = getData(f"{STOCK}", trainStart, trainEnd)
    histTestIndex = getData(f"{STOCK}", testStart, testEnd)
    trainY = getYnumpy(histTrainIndex)
    testY = getYnumpy(histTestIndex)
    print(f"target shape train: {len(trainY)}")
    print(f"target shape test: {len(testY)}")

    # print(trainY)

    saveDataSet(dataPath, trainX, trainY, testX, testY)

#train it with CNN
def train():
    classes = ["sell", "buy"]
    trainX, trainY, testX, testY = loadDataSet(dataPath)

    # numStocks = getNumStocks(testX, trainX)
    cnn = CNN((intervalPeriod, 1))

    global results
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

#make predictions on old data
def test():
    trainX, trainY, testX, testY = loadDataSet(dataPath)

    if NEW_MODEL:
        bestModel = tf.keras.models.load_model(checkpointPath)
    else:
        bestModel = tf.keras.models.load_model(previousSavePath)

    print(f"[INFO] Making Predictions.")
    predictions = bestModel.predict(testX)
    print(predictions)

    evalPreds(predictions, testY)

    # histTestIndex = getData(f"{STOCK}", testStart, testEnd)
    # histTestIndex = histTestIndex.iloc[daysBefore+daysAhead-1:]
    fig = getLossGraph(results)
    plt.savefig(graphPath)
    plt.show()

    #also make graph of predictions vs testY? and can plot last momentum vs if correct sell or buy

    #ask to save model if new model
    if NEW_MODEL:
        keep = askUserSaveModel()
        if keep == "y":
            version = getVersionName()

            newFolderPath = makeNewFolder(version)
            # saveIncludedStocks(newFolderPath)
            saveModel(newFolderPath, bestModel)
            saveParameters(newFolderPath, version)

#predict future date
def PredictOnDate():
    daysBefore, daysAhead = parseDaysBeforeAndAhead()

    predictStart = getDateInPast(predictDate,daysAhead+daysBefore-1)
    print(f"predict start date: {predictStart}")
    predictEnd = getDateInPast(predictDate, daysAhead-1)
    print(f"predict end date: {predictEnd}")

    if NEW_MODEL:
        stocksIncluded = readFile(stocksIncludedPath)
    else:
        stocksIncluded = readFile(previousStocksIncludedPath)

    print("[INFO] Loading Prediction Datasets.")

    INDEX_STOCKS = getTickers()

    #loads necessary data for the one needed prediction
    stockHistsTestX = []
    for stock in INDEX_STOCKS:
        if " " + stock + " " in stocksIncluded:
            print(f"[INFO] Loading Testset For {stock}.")
            test = getData(f"{stock}", predictStart, predictEnd)
            test = getXnumpyPredict(test)
            test = preprocessing.normalize(test)
            testXstock = np.transpose(test)
            print(f"test stock shape: {testXstock.shape}")

            #remove NaN from dataset
            # testXstock = removeNaNsingleColumn(testXstock)

            #adds or removes rows as needed
            if testXstock.shape[0] != daysBefore:
                stockArray = fixDataSingleColumn(stockArray, daysBefore)

            testXstock = testXstock.reshape((1,daysBefore,-1))
            stockHistsTestX.append(testXstock)

    print(f"[INFO] Rehaping Testset.")
    print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")

    testX = stackTransposeSwap(stockHistsTestX)
    print(f"total shape after change test: {testX.shape}")

    # pt = preprocessing.PowerTransformer()
    # testX = pt.fit_transform(testX)

    # testX = np.log(testX)

    if NEW_MODEL:
        bestModel = tf.keras.models.load_model(checkpointPath)
    else:
        bestModel = tf.keras.models.load_model(previousSavePath)

    print(f"[INFO] Making Prediction.")
    prediction = bestModel.predict(testX)

    print(f"\nprediction for {predictDate}: {prediction[0][0]}\n")

def main():
    setModes()
    setCustomCallback()

    if LOAD_DATASET:
        loadData()
    if TRAIN:
        train()
    if TEST:
        test()
    if PREDICT_ON_DATE:
        PredictOnDate()

main()
