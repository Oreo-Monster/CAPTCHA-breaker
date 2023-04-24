'''
Kura Yamada and Eda Wright

CNN for alphanumeric classification from a CAPTCHA data set

'''
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

class Single_Char_Net:

    def __init__(self, M, lr=0.001):
        self.build_model(M, lr)


    def build_model(self, M, lr):
        #Builds a CNN classifier for M classes
        #Single character recognition
        self.net = models.Sequential()
        
        #CNN layer
        self.net.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,1)))
        self.net.add(layers.MaxPooling2D((2, 2)))
        self.net.add(layers.Conv2D(60, (5, 5), activation='relu'))
        self.net.add(layers.MaxPooling2D((2, 2)))
        self.net.add(layers.Conv2D(64, (3, 3), activation='relu'))

        #Dense Layer
        self.net.add(layers.Flatten())
        self.net.add(layers.Dense(128, activation='relu'))
        self.net.add(layers.Dense(34, activation='softmax'))

        self.net.compile(optimizer=optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'], run_eagerly=True)


    def train(self, x_train, y_train, x_val, y_val, epochs=12, mini_batch=32):
        #Encoding words in dictionary
        self.ind2char = {}
        self.char2ind = {}
        for i, char in enumerate(np.unique(y_train)):
            self.ind2char[i] = char
            self.char2ind[char] = i
        y_train_ind = np.vectorize(self.char2ind.get)(y_train)
        y_val_ind = np.vectorize(self.char2ind.get)(y_val)

        return self.net.fit(x_train, y_train_ind, epochs=12, 
                            validation_data=(x_val, y_val_ind), batch_size=mini_batch)

    def predict(self, x, verbose=False):
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        net_acts = self.net.predict(x, verbose=verbose)
        y = np.argmax(net_acts,axis=1)
        return np.vectorize(self.ind2char.get)(y)
