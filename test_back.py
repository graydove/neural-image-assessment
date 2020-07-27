import numpy as np
import argparse
import sys
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.score_utils import mean_score, std_score

img = []
img.append(sys.argv[1])
target_size = (224, 224)

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('/home/graydove/Graydove/NIMA/weights/mobilenet_weights.h5')

    for img_path in img:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

    scores = [float('%.2f' % i) for i in scores]
    total = sum(np.array(scores) * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    if total > 5.5:
        scores = np.array(scores)*np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scores = list(scores)
    else:
        scores = np.array(scores)*np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scores = list(scores)
    scores = [float('%.2f' % i) for i in scores]
    print str('[result]:') + str(scores)