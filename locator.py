'''
Kura and Eda

Runs locator network on image.
'''
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tqdm import tqdm

class Locator_Net():

    def __init__(self, lr=0.001):
        self.segnet = models.Sequential()
        #CNN layer
        self.segnet.add(layers.Conv2D(6, (5,5), activation='relu', input_shape=(32,32,1)))
        self.segnet.add(layers.MaxPooling2D((2, 2)))
        self.segnet.add(layers.Conv2D(16, (5, 5), activation='relu'))
        self.segnet.add(layers.MaxPooling2D((2, 2)))

        #Dense Layer
        self.segnet.add(layers.Flatten())
        self.segnet.add(layers.Dense(32, activation='relu'))
        self.segnet.add(layers.Dense(1, activation='sigmoid'))
        self.segnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=10):
        history = self.segnet.fit(x_train, y_train, epochs=epochs, 
                    validation_data=(x_val, y_val), batch_size=batch_size)
        return history
    
    #Setting up filter function (Taken from CNN Project)
    def predict(self, img, frame=(32,32), verbose=False):
        img_y, img_x, n_chan = img.shape
        ker_x, ker_y = frame

        if verbose:
            print(f'img_x={img_y}, img_y={img_x}')
            print(f'ker_x={ker_x}, ker_y={ker_y}')

        if ker_x != ker_y:
            print('Kernels must be square!')
            return

        padding = int(np.ceil((ker_x - 1)/2))

        #Output array
        filteredImg = np.zeros(img.shape)

        num_iter = (img_y-2*padding) * (img_x-2*padding)
        if verbose:
            print(f"Begening locator on {num_iter} pixels")

        row = np.zeros(((img_x-2*padding), ker_y, ker_x, n_chan))

        for i in tqdm (range(padding, img_y-padding), desc="Scanning Image..."):
            #Collect all frames from a row into memory    
            for j in range(padding, img_x-padding):
                correction = 0 if ker_x%2 == 0 else 1
                row[j-padding,:,:,:] = img[i-padding:i+padding+correction, j-padding:j+padding+correction, :]
                
            #feed row into network for predictions (hopefully with parellel proc.)        
            filteredImg[i, padding:img_x-padding, :] = self.segnet.predict(row, verbose=0)

        return filteredImg
