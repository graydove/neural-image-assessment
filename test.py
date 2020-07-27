import numpy as np
import argparse
import sys
import os
# from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from utils.score_utils import mean_score, std_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return(files)  


# file_dir = "C:/Users/grayd/Desktop/NIMA/"
# test = file_name(file_dir)
# all_test = [file_dir + i for i in test]
all_test = ['C:/Users/grayd/Desktop/NIMA/954022.jpg']

# parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
# parser.add_argument('-dir', type=str, default=None,
                    # help='Pass a directory to evaluate the images in it')

# parser.add_argument('-img', type=str, default=[None], nargs='+',
                    # help='Pass one or more image paths to evaluate them')

# parser.add_argument('-resize', type=str, default='false',
                    # help='Resize images to 224x224 before scoring')

# parser.add_argument('-rank', type=str, default='true',
                    # help='Whether to tank the images after they have been scored')

# args = parser.parse_args()
# resize_image = args.resize.lower() in ("true", "yes", "t", "1")
target_size = (224, 224) # if resize_image else None
# rank_images = args.rank.lower() in ("true", "yes", "t", "1")

# give priority to directory
# if args.dir is not None:
    # print("Loading images from directory : ", args.dir)
    # imgs = Path(args.dir).files('*.png')
    # imgs += Path(args.dir).files('*.jpg')
    # imgs += Path(args.dir).files('*.jpeg')

# elif args.img[0] is not None:
    # print("Loading images from path(s) : ", args.img)
    # imgs = args.img

img = all_test
# img = []
# img.append("/home/ubuntu/Nvme/AVA_test/test/547917.jpg")
# img.append(sys.argv[1])


# else:
    # raise RuntimeError('Either -dir or -img arguments must be passed as argument')



base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.load_weights('C:/Users/grayd/Desktop/NIMA/weights/mobilenet_test.h5')
# model.save("/home/graydove/Graydove/NIMA/abc.h5")
# score_list = []
# testfile = open("/home/ubuntu/Nvme/AVA_test/testv1.txt", "a+")
nu = 0

for img_path in img:
    img = load_img(img_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    scores = model.predict(x, batch_size=1, verbose=0)[0]

    mean = mean_score(scores)
    std = std_score(scores)

    # file_name = Path(img_path).name.lower()
    # score_list.append((file_name, mean))

    # print("Evaluating : ", img_path)
    nu += 1
    # print(sc)
    # testfile.write(str(img_path.split('/')[-1]) + '\t' + str(mean) + '\n')

    # print()
    print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
    # print()

# if rank_images:
    # print("*" * 40, "Ranking Images", "*" * 40)
    # score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

    # for i, (name, score) in enumerate(score_list):
        # print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))


