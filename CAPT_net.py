'''
Model trained to break 5 charecter B&W CAPTCHAS

1. Uses locator CNN and K means clustering to find the
position of each letter.

2. A single charecter classifier run on each letter


'''

from locator import Locator_Net
from single_char_net import Single_Char_Net
from sklearn.cluster import MiniBatchKMeans
import numpy as np

class CAPT_NET:

    def __init__(self,K,M,verbose=False, loc_lr=0.0015, class_lr = 0.001):
        self.verbose = verbose
        self.K = K
        self.loc_net = Locator_Net(lr=loc_lr)
        self.class_net = Single_Char_Net(M, class_lr)
        self.kmeans = MiniBatchKMeans(n_clusters = K, n_init=10)

    def train_loc(self,x_train,y_train, x_val, y_val, loc_mini_batch=100, loc_epochs=5):
        return self.loc_net.train(x_train, y_train, x_val, y_val, batch_size=loc_mini_batch, epochs = loc_epochs)

    def train_class(self, x_train, y_train, x_val, y_val, epochs, mini_batch):
        return self.class_net.train(x_train, y_train, x_val, y_val, epochs, mini_batch)

    def cluster(self, heat_map):
        threshold = np.squeeze(np.where(heat_map > 0.5, 1, 0)) 
        idx_row, idx_col = np.nonzero(threshold)
        non_zeros = list(zip(idx_row, idx_col)) 
        self.kmeans.fit(non_zeros)
        return self.kmeans.cluster_centers_

    def predict(self, x):
        heat_map = self.loc_net.predict(x, verbose = self.verbose)
        centriods = self.cluster(heat_map)
        centriods.sort(axis=0)
        prediction = ""
        for i in range(self.K):
            cent = centriods[i,:]
            cent0 = int(cent[0])
            cent1 = int(cent[1])
            cropped = x[cent0-16:cent0+16 , cent1-16:cent1+16,:]
            print(cropped.shape)

            prediction += self.class_net.predict(cropped)[0]

        return prediction, heat_map, centriods


