import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, Callback

from utils.data_loader import train_generator, val_generator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存


'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
	def __init__(self, *args, **kwargs):
		super(TensorBoardBatch, self).__init__(*args)

		# conditionally import tensorflow iff TensorBoardBatch is created
		self.tf = __import__('tensorflow')

	def on_batch_end(self, batch, logs=None):
		logs = logs or {}

		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			summary = self.tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, batch)

		self.writer.flush()

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			summary = self.tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, epoch * self.batch_size)

		self.writer.flush()

def earth_mover_loss(y_true, y_pred):
	cdf_ytrue = K.cumsum(y_true, axis=-1)
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())  
		print(cdf_ytrue)
		
	print(cdf_ytrue)
	cdf_ypred = K.cumsum(y_pred, axis=-1)
	print(cdf_ypred)
	samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
	# print(K.mean(samplewise_emd))
	return K.mean(samplewise_emd)
	
def step_decay(epoch, lr):
    # initial_lrate = 1.0 # no longer needed
    boundaries = [7,20] # 为不同的epoch范围设置不同的学习率
    values = [0.95*3e-6, 0.95*0.95*3e-6]
    for idx,bd in enumerate(boundaries):
        if (epoch+1)<bd:
            lr = values[idx]
            print(epoch, lr)
            return lr
    print(epoch)
    return values[-1]


class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)
	

image_size = 224

base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
for layer in base_model.layers:
	layer.trainable = True

x = Dropout(0.75)(base_model.output)
x = SetLearningRate(Dense(10, activation='softmax'), 10, True)(x)

model = Model(base_model.input, x)
model.summary()
optimizer = Adam(lr=3e-7, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(optimizer, loss=earth_mover_loss)

# load weights from trained model if it exists
if os.path.exists('weights/mobilenet_test.h5'):
	model.load_weights('weights/mobilenet_test.h5')

checkpoint = ModelCheckpoint('weights/mobilenet_test.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard, LearningRateScheduler(step_decay)]

batchsize = 100
epochs = 20

model.fit_generator(train_generator(batchsize=batchsize),
					steps_per_epoch=(225000 // batchsize),
					epochs=epochs, verbose=1, callbacks=callbacks,
					validation_data=val_generator(batchsize=batchsize),
					validation_steps=(5000 // batchsize))