#STUFF TO DO
#work on architecture
#convert price to price gains?

print('[INFO] Importing packages.')
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yfinance import *
from datetime import *
from os import mkdir

from parameters import *        #edit parameters from parameters.py
from cnn import *               #edit cnn from cnn.py

print('[INFO] Done importing packages.')

#checks if GPU is recognized
def checkGPU():
    global devices
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 0:
        print('[INFO] GPU is detected.')
    else:
        print('[INFO] GPU not detected.')
#overides load, train, test, when predicting or using quick run
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
    OHLCcolumn = histNP[OHLC]
    OHLCcolumn = removeNaNall(OHLCcolumn)

    # OHLCcolumn = preprocessing.normalize(OHLCcolumn)

    # pt = preprocessing.PowerTransformer()
    # OHLCcolumn = pt.fit_transform([OHLCcolumn])

    X = []
    #uses daysBefore previous days
    for x in range(len(OHLCcolumn)-daysBefore-daysAhead+1):
        X.append(OHLCcolumn[x:x+daysBefore])

    X = np.array(X)

    return X
#get a formatted dataset of OHLC of stock as numpy array
def getXnumpyPredict(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    OHLCcolumn = histNP[OHLC]
    OHLCcolumn = removeNaNall(OHLCcolumn)

    # pt = preprocessing.PowerTransformer()
    # OHLCcolumn = pt.fit_transform(OHLCcolumn)

    X = np.array([OHLCcolumn])

    return X
#get a formatted dataset of OHLC of index as numpy array
def getYnumpy(hist):
    histNP = hist.to_numpy()
    #to get columns
    histNP = np.transpose(histNP)
    OHLCcolumn = histNP[OHLC]
    OHLCcolumn = removeNaNall(OHLCcolumn)

    Y = []
    #target is daysBefore+1
    for y in range(len(OHLCcolumn)-daysBefore-daysAhead+1):
        Y.append(OHLCcolumn[y+daysBefore+daysAhead-1])

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
                print("[Error] NaN Found")
                stockArray[rowIndex][i] = avg

    return stockArray
#locate and remove NaN in 1 dimensional numpy array
def removeNaNall(array):
    for i in range(array.shape[0]):
        # print(array.shape[0])
        if np.isnan(array[i]):
            print("[Error] NaN Found")
            if i==0:
                array[i] = array[i+1]
                continue
            if i==array.shape[0]-1:
                array[i] = array[i-1]
                continue
            array[i] = (array[i+1] + array[i-1]) / 2

    return array
#same as above but for single column of data
def removeNaNsingleColumn(stockArray):
    sum = 0
    counter = 0
    for val in stockArray:
        if not np.isnan(val[0]):
            sum += val[0]
            counter += 1
    avg = sum/counter

    for index in range(stockArray.shape[0]):
        if np.isnan(stockArray[index][0]):
            print("[Error] NaN Found")
            stockArray[index][0] = avg

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
#same as above but for single column
def fixDataSingleColumn(stockArray, expectedVal):
    while stockArray.shape[0] < expectedVal:
        print(stockArray)
        stockArray = np.vstack([stockArray[0],stockArray])
        print(stockArray)
    while stockArray.shape[0] > expectedVal:
        print(stockArray)
        stockArray = np.delete(stockArray,0)
        print(stockArray)

    return stockArray
#vstack, transpose, swaps axis 1 and 2
def stackTransposeSwap(stocksList):
    stocksCombined = np.vstack(stocksList)
    xSet = np.transpose(stocksCombined)
    xSet = np.swapaxes(xSet,1,2)

    return xSet

#get number of stocks used in dataset
def getNumStocks(testX, trainX, holdoutX):
    assert trainX.shape[1] == testX.shape[1] and trainX.shape[1] == holdoutX.shape[1]
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
def saveDataSet(dataPath, trainX, trainY, testX, testY, holdoutX, holdoutY):
    with open(dataPath, 'wb') as f:
        np.save(f, trainX)
        np.save(f, trainY)
        np.save(f, testX)
        np.save(f, testY)
        np.save(f, holdoutX)
        np.save(f, holdoutY)
#loads numpy arrays from npy file
def loadDataSet(dataPath):
    with open(dataPath, 'rb') as f:
        trainX = np.load(f)
        trainY = np.load(f)
        testX = np.load(f)
        testY = np.load(f)
        holdoutX = np.load(f)
        holdoutY = np.load(f)

    return trainX, trainY, testX, testY, holdoutX, holdoutY

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

#displays prediction with  date followed by prediction
def displayPredictionsAsText(histIndex, predictions):
    histIndex.index = histIndex.index.date

    counter = 0
    for date in histIndex.index:
        print(f"{date}: {round(predictions[counter][0],2)}")
        counter+=1

#ask user if they want to save the model
def askUserSaveModel():
    if remoteMachine:
        return "y"

    else:
        while True:
            keep = input("save this model to folder? (y/n)")
            if keep != "y" and keep != "n":
                print("error")
                continue
            break

        return keep
#ask user for version name
def getVersionName():
    if remoteMachine:
        return remoteVersionName

    else:
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
    newFolderPath = f"{savedModelsPath}/{daysBefore}_{daysAhead}_{OHLC}_{version}"
    mkdir(newFolderPath)

    return newFolderPath
#saves pyplot to folder for later analysis
def savePyPlot(newFolderPath, version, holdoutItems, testItems, trainItems):
    print("[INFO] Saving Pyplot.")
    fig = getLossAndPriceGraph(results, holdoutItems, testItems, trainItems)
    plt.savefig(f"{newFolderPath}/{daysBefore}_{daysAhead}_{OHLC}_{version}.png")
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
    f = open(f"{newFolderPath}/{daysBefore}_{daysAhead}_{OHLC}_{version}_info.txt", 'w')
    f.write(f"version name: {daysBefore}_{daysAhead}_{OHLC}_{version}\n")
    f.write(f"training dates: {trainStart} to {trainEnd}\n")
    f.write(f"testing dates: {testStart} to {testEnd}\n")
    f.write(f"holdout dates: {holdoutStart} to {holdoutEnd}\n")
    f.write(f"days before: {daysBefore}\n")
    f.write(f"days ahead: {daysAhead}\n")
    f.write(f"OHLC column: {OHLC}\n")
    f.write(f"number of stocks included: {numStocks}\n")
    f.close()

#get loss and price graph
def getLossAndPriceGraph(results, holdoutItems, testItems, trainItems):
    #index 0 is dates, index 1 is real, index 2 is predictions
    fig = plt.figure("preds vs real high price", figsize=(15, 8))
    fig.tight_layout()
    #training and validation loss
    plt1 = fig.add_subplot(221)
    plt1.title.set_text("training and validation loss")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['loss'], color="green", label="training")
    plt1.plot(np.arange(0, TRAIN_EPOCHS), results.history['val_loss'], color="red", label="validation")
    plt1.legend(loc='upper right')
    #training set
    plt2 = fig.add_subplot(222)
    plt2.title.set_text("training set real and preds")
    plt2.plot(trainItems[0], trainItems[1], color="blue", label="real")
    plt2.plot(trainItems[0], trainItems[2], color="red", label="preds")
    plt2.legend(loc='upper left')
    #validation set
    plt3 = fig.add_subplot(223)
    plt3.title.set_text("validation set real and preds")
    plt3.plot(testItems[0], testItems[1], color="blue", label="real")
    plt3.plot(testItems[0], testItems[2], color="red", label="preds")
    plt3.legend(loc='upper left')
    #holdout set
    plt4 = fig.add_subplot(224)
    plt4.title.set_text("holdout set real and preds")
    plt4.plot(holdoutItems[0], holdoutItems[1], color="blue", label="real")
    plt4.plot(holdoutItems[0], holdoutItems[2], color="red", label="preds")
    plt4.legend(loc='upper left')

    return fig
#get real vs preds price graph
def getJustPriceGraph(holdoutItems, testItems):
    fig = plt.figure("preds vs real high price", figsize=(10, 8))
    fig.tight_layout()
    #validation set
    plt1 = fig.add_subplot(211)
    plt1.title.set_text("validation set real and preds")
    plt1.plot(testItems[0], testItems[1], color="blue", label="real")
    plt1.plot(testItems[0], testItems[2], color="red", label="preds")
    plt1.legend(loc='upper left')
    #holdout set
    plt2 = fig.add_subplot(212)
    plt2.title.set_text("holdout set real and preds")
    plt2.plot(holdoutItems[0], holdoutItems[1], color="blue", label="real")
    plt2.plot(holdoutItems[0], holdoutItems[2], color="red", label="preds")
    plt2.legend(loc='upper left')

    return fig

#get days before and ahead from saved model name
def parseDaysBeforeAndAhead():
    index1 = savedModelName.find("_")
    daysBefore = int(savedModelName[:index1])
    print(f"parsed daysBefore: {daysBefore}")
    savedModelName2 = savedModelName[index1+1:]
    index2 = savedModelName2.find("_")
    daysAhead = int(savedModelName2[:index2])
    print(f"parsed daysAhead: {daysAhead}")

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

#load and format dataset
def loadData():
    print("[INFO] Loading Traning and Test Datasets.")

    #index target prices
    print("\n[INFO] Loading Index Data.")
    histTrainIndex = getData(f"{INDEX}", trainStart, trainEnd)
    histTestIndex = getData(f"{INDEX}", testStart, testEnd)
    histHoldoutIndex = getData(f"{INDEX}", holdoutStart, holdoutEnd)
    trainY = getYnumpy(histTrainIndex)
    testY = getYnumpy(histTestIndex)
    holdoutY = getYnumpy(histHoldoutIndex)
    print(f"target shape train: {trainY.shape[0]}")
    print(f"target shape test: {testY.shape[0]}")
    print(f"target shape holdout: {holdoutY.shape[0]}")

    expectedTrain = trainY.shape[0]
    expectedTest = testY.shape[0]
    expectedHoldout = holdoutY.shape[0]

    #uses all 500+ stocks in the index
    if USE_ALL_STOCKS:
        INDEX_STOCKS = getTickers()
    #uses just 10 stocks for faster testing
    else:
        INDEX_STOCKS = shortList

    f = open(f"{stocksIncludedPath}", 'w')

    stockHistsTrainX = []
    stockHistsTestX = []
    stockHistsHoldoutX = []

    for stock in INDEX_STOCKS:
        print(f"\n[INFO] Loading Dataset For {stock}.")
        train = getData(f"{stock}", trainStart, trainEnd)
        test = getData(f"{stock}", testStart, testEnd)
        holdout = getData(f"{stock}", holdoutStart, holdoutEnd)

        #converts the hist data into formatted numpy array
        #this is where most of the data shaping happens
        train = getXnumpy(train)
        test = getXnumpy(test)
        holdout = getXnumpy(holdout)

        # if train.shape[0] != 0 and test.shape[0] != 0:
        #     train = preprocessing.normalize(train)
        #     test = preprocessing.normalize(test)
        # train = removeNaN(getXnumpy(train))
        # test = removeNaN(getXnumpy(test))

        trainXstock = np.transpose(train)
        testXstock = np.transpose(test)
        holdoutXstock = np.transpose(holdout)

        #error with diviison by zero
        # trainXstock = pt.fit_transform(trainXstock)
        # testXstock = pt.fit_transform(testXstock)

        print(f"train stock shape: {trainXstock.shape}")
        print(f"test stock shape: {testXstock.shape}")
        print(f"holdout stock shape: {holdoutXstock.shape}")

        print(f"[INFO] Checking stock validity.")
        #will only attempt to add stock to the array if it exists
        if trainXstock.shape[0] != 0 and testXstock.shape[0] != 0 and holdoutXstock.shape[0] != 0:
            if trainXstock.shape[1] != expectedTrain or testXstock.shape[1] != expectedTest or holdoutXstock.shape[1] != expectedHoldout:
                print("[Error] Possibly did not set expectedTrain and expectedTest")
                print("[Fix] Replace expected values in parameters.py to match above values")

                #fix data to match expected shapes
                trainXstock = fixData(trainXstock, expectedTrain)
                testXstock = fixData(testXstock, expectedTest)
                holdoutXstock = fixData(holdoutXstock, expectedHoldout)

                print("[INFO] Sketch fix, revised array shape below")
                print(f"train stock shape: {trainXstock.shape}")
                print(f"test stock shape: {testXstock.shape}")
                print(f"holdout stock shape: {holdoutXstock.shape}")

            if trainXstock.shape[1] == expectedTrain and testXstock.shape[1] == expectedTest and holdoutXstock.shape[1] == expectedHoldout:
                print(f"[INFO] Stock valid.")
                #write stock name to list of included stocks
                f.write(f" {stock} ")

                #reshape stock for easier combination later
                trainXstock = trainXstock.reshape((1,daysBefore,-1))
                testXstock = testXstock.reshape((1,daysBefore,-1))
                holdoutXstock = holdoutXstock.reshape((1,daysBefore,-1))

                #add individual stock to array of total stocks in index
                stockHistsTrainX.append(trainXstock)
                stockHistsTestX.append(testXstock)
                stockHistsHoldoutX.append(holdoutXstock)

        else:
            print(f"[Error] Stock invalid.")

    f.close()

    print(f"[INFO] Rehaping Dataset.")
    # print(f"train stock shape after reshape: {stockHistsTrainX[0].shape}")
    # print(f"test stock shape after reshape: {stockHistsTestX[0].shape}")
    trainX = stackTransposeSwap(stockHistsTrainX)
    testX = stackTransposeSwap(stockHistsTestX)
    holdoutX = stackTransposeSwap(stockHistsHoldoutX)

    # pt = preprocessing.PowerTransformer()
    # trainX = pt.fit_transform(trainX)
    # testX = pt.fit_transform(testX)

    print(f"total trainX shape after reshape: {trainX.shape}")
    print(f"total testX shape after reshape: {testX.shape}")
    print(f"total holdoutX shape after reshape: {holdoutX.shape}")

    print(f"target shape train: {trainY.shape[0]}")
    print(f"target shape test: {testY.shape[0]}")
    print(f"target shape holdout: {holdoutY.shape[0]}")

    saveDataSet(dataPath, trainX, trainY, testX, testY, holdoutX, holdoutY)

#train model with CNN
def train():
    trainX, trainY, testX, testY, holdoutX, holdoutY = loadDataSet(dataPath)

    # trainX = np.log(trainX)
    # testX = np.log(testX)

    numStocks = getNumStocks(testX, trainX, holdoutX)
    print(numStocks)
    print(daysBefore)
    cnn = CNN((numStocks, daysBefore))

    print("[INFO] Printing Tensorflow CNN Summary...")
    print(cnn)

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
    trainX, trainY, testX, testY, holdoutX, holdoutY = loadDataSet(dataPath)

    # trainX = np.log(trainX)
    # testX = np.log(testX)

    #test the model on the holdout set
    histHoldoutIndex = getData(f"{INDEX}", holdoutStart, holdoutEnd)
    histHoldoutIndex = histHoldoutIndex.iloc[daysBefore+daysAhead-1:].index

    #as well as the test set
    histTestIndex = getData(f"{INDEX}", testStart, testEnd)
    histTestIndex = histTestIndex.iloc[daysBefore+daysAhead-1:].index

    #as well as the validation set
    histTrainIndex = getData(f"{INDEX}", trainStart, trainEnd)
    histTrainIndex = histTrainIndex.iloc[daysBefore+daysAhead-1:].index

    #for testing on a new model
    if NEW_MODEL:
        bestModel = tf.keras.models.load_model(checkpointPath)
    #for testing on a previously saved model
    else:
        bestModel = tf.keras.models.load_model(previousSavePath)

    print(f"[INFO] Making Predictions.")
    holdoutPredictions = bestModel.predict(holdoutX)
    testPredictions = bestModel.predict(testX)
    trainPredictions = bestModel.predict(trainX)

    # displayPredictionsAsText(histHoldoutIndex, predictions)

    holdoutItems = [histHoldoutIndex, holdoutY, holdoutPredictions]
    testItems = [histTestIndex, testY, testPredictions]
    trainItems = [histTrainIndex, trainY, trainPredictions]

    if TRAIN:
        fig = getLossAndPriceGraph(results, holdoutItems, testItems, trainItems)
        plt.savefig(graphPath)
    else:
        fig = getJustPriceGraph(holdoutItems, testItems)
        plt.savefig(graphPath)

    if not remoteMachine:
        plt.show()
        print(f"[INFO] Close plot to continue.")

    #ask to save model if new model
    if NEW_MODEL:
        keep = askUserSaveModel()
        if keep == "y":
            version = getVersionName()

            newFolderPath = makeNewFolder(version)
            savePyPlot(newFolderPath, version, holdoutItems, testItems, trainItems)
            saveIncludedStocks(newFolderPath)
            saveModel(newFolderPath, bestModel)

            numStocks = getNumStocks(testX, trainX, holdoutX)
            saveParameters(newFolderPath, version, numStocks)

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
        previousStocksIncludedPath = f"{previousSavePath}/stocksIncluded.txt"
        stocksIncluded = readFile(previousStocksIncludedPath)

    print("[INFO] Loading Prediction Datasets.")

    INDEX_STOCKS = []
    temp = ""
    listen = False
    for char in stocksIncluded:
        if listen:
            if char != " ":
                temp += char
            else:
                INDEX_STOCKS.append(temp.strip())
                temp = ""
                listen = False
        else:
            if char != " ":
                temp += char
                listen = True

    # print(len(INDEX_STOCKS))

    # INDEX_STOCKS = getTickers()

    #loads necessary data for the one needed prediction
    stockHistsTestX = []
    for stock in INDEX_STOCKS:
        print(f"\n[INFO] Loading Testset For {stock}.")
        test = getData(f"{stock}", predictStart, predictEnd)
        test = getXnumpyPredict(test)
        # test = preprocessing.normalize(test)
        testXstock = np.transpose(test)
        print(f"test stock shape: {testXstock.shape}")

        #remove NaN from dataset
        # testXstock = removeNaNsingleColumn(testXstock)

        #adds or removes rows as needed
        if testXstock.shape[0] != daysBefore:
            testXstock = fixDataSingleColumn(testXstock, daysBefore)

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
    checkGPU()
    setModes()
    setCustomCallback()

    global OHLC
    for val in OHLCvals:
        OHLC = val

        if LOAD_DATASET:
            loadData()
        if TRAIN:
            train()
        if TEST:
            test()
        if PREDICT_ON_DATE:
            PredictOnDate()

if __name__ == "__main__":
    main()
