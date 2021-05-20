import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dropout
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

TRAIN_EPOCHS = 100
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

#CNN for 3D numpy array
class CNN():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        # For Conv2D, you give it: Outgoing Layers, Frame size.  Everything else needs a keyword.
        # Popular keyword choices: strides (default is strides=1), padding (="valid" means 0, ="same" means whatever gives same output width/height as input).  Not sure yet what to do if you want some other padding.
        # Activation function is built right into the Conv2D function as a keyword argument.

        self.model.add(layers.Conv1D(16, 3, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.BatchNormalization(trainable=False))
        self.model.add(Dropout(0.05))

        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        # self.model.add(layers.Conv1D(64, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.1))
        #
        # self.model.add(layers.Conv1D(128, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.15))
        #
        # self.model.add(layers.Conv1D(256, 3, activation = 'relu'))
        # self.model.add(layers.BatchNormalization(trainable=False))
        # self.model.add(Dropout(0.2))

        # self.model.add(layers.MaxPooling2D(pool_size = 2))

        self.model.add(layers.Flatten())

        #get to one value
        # self.model.add(layers.Dense(2400, activation = 'relu', input_shape = input_shape))
        # self.model.add(layers.Dense(1200, activation = 'relu'))
        # self.model.add(layers.Dense(600, activation = 'relu'))
        # self.model.add(layers.Dense(300, activation = 'relu'))
        # self.model.add(layers.Dense(120, activation = 'relu'))
        # self.model.add(layers.Dense(60, activation = 'relu'))
        # self.model.add(layers.Dense(20, activation = 'relu'))
        # self.model.add(layers.Dense(1))

        self.model.add(layers.Dense(32, activation = 'relu', input_shape = input_shape))
        self.model.add(layers.Dense(16, activation = 'relu'))
        self.model.add(layers.Dense(1))

        #lr=0.001, momentum=0.9
        self.optimizer = optimizers.Adam(lr=0.00001)
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
