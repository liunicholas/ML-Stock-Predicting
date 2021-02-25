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

def getData(stockName):
    stock = Ticker(stockName)
    hist = stock.history(period="1mo")

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

def main():
    ticker = "TSLA"
    hist = getData(ticker)

    displayStock(hist, ticker)

main()
