import _pickle as cPickle

import cv2
import numpy as np

encs_test_label = np.ones(1)
encs_train_label = -np.ones(9)
y_labels = np.concatenate((encs_test_label, encs_train_label), axis=0)
print(y_labels.shape)




