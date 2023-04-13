'''
Kura Yamada and Eda Wright

Preprocesses images generated from claptcha
    - converts to gray scale
    - noramlizes
    - Shuffles
    - Adds singelton dims

Preprocessed data is saved to disk (expensive to preprocess)

Can also split data and return them as arrays
'''

import numpy as np

def preprocess_image_data(path_to_x, path_to_y):
    x = np.load(path_to_x)
    y = np.load(path_to_y)
    #Converting to greyscale
    x = np.mean(x, axis=3)
    #Normalizing
    x /= 256
    #Shuffling
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx, :, :]
    y = y[idx]
    x = np.squeeze(x)
    if(len(x.shape) == 3):
        x = x[:, :, :, None]
    
    np.save(f"{path_to_x[:-4]}_preprocessed.npy", x)
    np.save(f"{path_to_y[:-4]}_preprocessed.npy", y)

    return x, y

def split(path_to_x, path_to_y, split_ratio=0.8, test_split=3000 ):
    '''Preprocess data first!'''
    x = np.load(path_to_x)
    y = np.load(path_to_y)
    N = x.shape[0]
    split_idx = int(N * split_ratio)
    test_split = int(test_split)
    x_train = x[:split_idx,:, :, :] 
    x_val = x[split_idx:-test_split, :, :, :]
    x_test = x[-test_split:, :, :, :]
    x_train_dev = x[:3000, :, :, :]
    x_val_dev = x[split_idx:split_idx+500, :, :, :]

    y_train = y[:split_idx] 
    y_val = y[split_idx:-test_split]
    y_test = y[-test_split:]
    y_train_dev = y[:3000]
    y_val_dev = y[split_idx:split_idx+500]

    return x_train, x_val, x_test, x_train_dev, x_val_dev, y_train, y_val, y_test, y_train_dev, y_val_dev
