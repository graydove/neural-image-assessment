import numpy as np
import argparse
import sys
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.score_utils import mean_score, std_score


def predict(img_path, model):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	preds = model.predict(x)
	return preds


def earth_mover_loss(y_true, y_pred):
	cdf_ytrue = K.cumsum(y_true, axis=-1)
	# print(cdf_ytrue)
	cdf_ypred = K.cumsum(y_pred, axis=-1)
	# print(cdf_ypred)
	samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
	# print(K.mean(samplewise_emd))
	return K.mean(samplewise_emd)


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

with tf.device('GPU'):
	base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
	x = Dropout(0.75)(base_model.output)
	x = Dense(10, activation='softmax')(x)

	model = Model(base_model.input, x)
	model.load_weights('weights/mobilenet_weights.h5')
	# t1 = time.time()
	# print('Loaded in:', t1-t0)

	# test_path = sys.argv[1]
	# print('Generating predictions on image:', sys.argv[2])
	# preds = predict(sys.argv[1], model)
	# preds = preds.tolist()[0][0][0]
	# result = preds.index(max(preds))

	# base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
	# x = Dropout(0.75)(base_model.output)
	# x = Dense(10, activation='softmax')(x)

	# model = Model(base_model.input, x)
	# model.load_weights('/home/graydove/Graydove/AVA_dataset_tag/MobileNet_2_class.h5')
	# # model.save("/home/graydove/Graydove/NIMA/abc.h5")
	# print(model)

	for img_path in img:
		print img_path
		preds = predict(img_path, model)
		preds = preds.tolist()[0]
		# print preds
		result = np.array([1,2,3,4,5,6,7,8,9,10])*np.array(preds)
		result = sum(result)
		print result
		# img = load_img(img_path, target_size=target_size)
		# x = img_to_array(img)
		# x = np.expand_dims(x, axis=0)
		# x = preprocess_input(x)
		#
		# scores = model.predict(x, batch_size=1, verbose=0)
		# print(score)
		# preds = scores.tolist()[0][0][0]
		# result = preds.index(max(preds))

		# mean = mean_score(scores)
		# # std = std_score(scores)

		# # cores = [float('%.2f' % i) for i in scores]
		predict_score[img_path.split('/')[-1].strip('.jpg')] = result
		# print img_path.split('/')[-1].strip('.jpg')
		# print result
		
		# print mean
		num += 1
		print num
		# print num

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

