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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from utils.score_utils import mean_score, std_score

f = open("/home/graydove/Datasets/AVA_dataset/AVA_test_name.txt", 'r')
img = []

for lines in f.readlines():
    img.append('/home/graydove/Datasets/AVA_dataset/images/' + lines.strip('\n').split(' ')[1] + '.jpg')

# g = open("/home/graydove/Datasets/AVA_dataset/AVA_predict.txt", 'a+')
real_score = {}
g = open("/home/graydove/Datasets/AVA_dataset/AVA_test.txt", 'r')
for lines in g.readlines():
    real_score[lines.strip('\n').split(' ')[0]] = float(lines.strip('\n').split(' ')[1])
# img.append(sys.argv[1])
target_size = (224, 224)

predict_score = {}
num = 0
with tf.device('/GPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('/home/graydove/Graydove/AVA_dataset_tag/MobileNet_EMD.h5')
    # model.save("/home/graydove/Graydove/NIMA/abc.h5")
    print(model)

    for img_path in img:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        # std = std_score(scores)

        # cores = [float('%.2f' % i) for i in scores]
        predict_score[img_path.split('/')[-1].strip('.jpg')] = mean
		
        # print scores
        print mean
        num += 1
        print num
		
        # print std	
accuracy = 0
for key in real_score:
    if real_score[key] > 5.5 and predict_score[key] > 5.5:
        accuracy += 1
    elif real_score[key] < 5.5 and predict_score[key] < 5.5:
        accuracy += 1
    else:
        print accuracy
print str(float('%.3f' % (accuracy/500))) + '%'
		
