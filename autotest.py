import numpy as np
import argparse
import sys
import os
import shutil
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.score_utils import mean_score, std_score
Path1 = sys.argv[1]
img = []
for root, _, files in os.walk(Path1, topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            if (os.path.splitext(name)[1] == ".jpg" ) or (os.path.splitext(name)[1] == ".png" ) :
                img.append(os.path.join(root, name))
result = {}
target_size = (224, 224)

with tf.device('/GPU:0'):
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
        fname,_=os.path.splitext(img_path)
        mean = mean_score(scores)
        result[img_path]=fname+" "+str(round(mean,2))
        
        std = std_score(scores)
        print img_path
        print mean
        print std
def anytest(Path,img,scores):
    
  	
    if not os.path.exists(Path):
        os.mkdir(Path)    
    print('res  '+Path)
    for i in scores:
    	
    	fname,_=os.path.splitext(str(result[i]))
    	print(fname)
        shutil.copyfile(i,os.path.join(Path,fname)+os.path.splitext(i)[1])
        f= open("result.txt",'a') 
        f.write(i+" "+str(result[i])+"\n")
        f.close()	

def autotest_rename(img,scores):
    
  	
    
    for i in scores:
    	
    	
    	os.rename(i,str(result[i])+os.path.splitext(i)[1])
        
        f= open("result.txt",'a') 
        f.write(i+" "+str(result[i])+"\n")
        f.close()	
autotest_rename(img,result)
    # scores = [float('%.2f' % i) for i in scores]
    # total = sum(np.array(scores) * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    # if total > 5.5:
        # scores = np.array(scores)*np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # scores = list(scores)
    # else:
        # scores = np.array(scores)*np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # scores = list(scores)
    # scores = [float('%.2f' % i) for i in scores]
    # print str('[result]:') + str(scores)
    # 
