'''
Kura and Eda

Runs locator network on image.

'''
import tensorflow as tf
from tensorflow.keras import layers, models

class Locator_Net():

    def __init__(self):
        self.segnet = models.Sequential()
        #CNN layer
        self.segnet.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)))
        self.segnet.add(layers.MaxPooling2D((2, 2)))
        self.segnet.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.segnet.add(layers.MaxPooling2D((2, 2)))
        self.segnet.add(layers.Conv2D(64, (3, 3), activation='relu'))

        #Dense Layer
        self.segnet.add(layers.Flatten())
        self.segnet.add(layers.Dense(64, activation='relu'))
        self.segnet.add(layers.Dense(1))
    
        self.segnet.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val):
        history = self.segnet.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_val, y_val))
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

    #     #Padding image with zeros
        padding = int(np.ceil((ker_x - 1)/2))
    #     img_padded = np.zeros((img_y+(2*padding), img_x+(2*padding), n_chan))
    #     img_padded[padding:-padding, padding:-padding, :] = img
    #     img_padded_y, img_padded_x, _ = img_padded.shape

    #     if verbose:
    #         print(f"Padded Image Size: {img_padded.shape}")
        #Output array
        filteredImg = np.zeros(img.shape)

        for i in range(padding, img_y-padding):
            for j in range(padding, img_x-padding):
                correction = 0 if ker_x%2 == 0 else 1
                selection = img[None, i-padding:i+padding+correction, j-padding:j+padding+correction, :]
                if verbose and i==padding and j==padding:
                    print(f"Frame size: {selection.shape}")
                filteredImg[i-padding, j-padding, :] = self.segnet.predict(selection)

        return filteredImg

    test = conv2_gray(x_multi[0], segnet, verbose=True)