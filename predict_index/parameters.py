remoteMachine = True                   #will automatically save the model
remoteVersionName = "remoteTestMay20"     #name of version for the saved model

#set QUICK_RUN to true for quick testing
#set PREDICT_ON_DATE to true and OVERRIDE to true for just predicting a date

QUICK_RUN = True              #for just testing code (OVERIDES EVERYTHING)

#dates for training and testing range
trainStart = "2005-01-01"
trainEnd = "2007-01-01"

testStart = "2008-01-01"
testEnd = "2010-01-01"

holdoutStart = "2011-01-01"
holdoutEnd = "2013-01-01"

LOAD_DATASET = True           #set to false when testing architecture
USE_ALL_STOCKS = True         #set to false for just testing
OHLC = 1                      #open = 0, high = 1, low = 2, close = 3

daysBefore = 10                #total days in period for prediction
daysAhead = 10                  #total days predicting in future

# expectedTrain = 2393            #find with test run
# expectedTest = 1576             #find with test run
# expectedHoldout = 1700           #find with test run

#edit epochs and batch sizes from cnn.py
TRAIN = True                  #makes the model

#tensorflow training variables
TRAIN_EPOCHS = 100
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

TEST = True                   #for comparing real with predictions
NEW_MODEL = True              #tests on a new model

#set both to true for purely predicitng date with specified model
PREDICT_ON_DATE = False        #set to true to predict day
OVERRIDE = False               #overrides load, train, test, and new_model

#vars for predicting
predictDate = "2021-05-20"
savedModelName = "10_10_remoteVersionTesting"

graphPath = "./info/pyplots/newestPlot.png"                  #save mpl graph
dataPath = "./info/datasets/allSpy.npy"                      #save npy arrays
checkpointPath = "./info/checkpoints"                        #save models
stocksIncludedPath = "./info/datasets/stocksIncluded.txt"    #save list of stocks used

savedModelsPath = "./savedModels"                            #save best model
# savedModelsPath = "/Volumes/transfer/indexModels"         #for using model from usb stick
previousSavePath = f"{savedModelsPath}/{savedModelName}"    #location of desired model for predicting

#the 9 federally recognized holidays
holidays2021 = ["2021-01-01", "2021-01-18", "2021-02-15",
    "2021-04-02", "2021-05-31", "2021-07-05",
    "2021-09-06", "2021-11-25", "2021-12-25"]

#focusing on SPDR S&P 500 ETF
INDEX = "SPY"
indexSource = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
shortList = ["AAPL", "MSFT", "AMZN", "FB", "GOOGL",
    "GOOG", "TSLA", "BRK.B", "JPM", "JNJ"]
