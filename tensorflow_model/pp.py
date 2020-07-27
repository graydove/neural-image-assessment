#encoding:utf-8
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import cv2


def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print tensors

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            op = sess.graph.get_operations()

          
            for m in op:
                print(m.values())

            input_x = sess.graph.get_tensor_by_name("convolution2d_1_input:0")  #具体名称看上一段代码的input.name
            print input_x

            out_softmax = sess.graph.get_tensor_by_name("activation_4/Softmax:0") #具体名称看上一段代码的output.name

            print out_softmax

            img = cv2.imread(jpg_path, 0)
            img_out_softmax = sess.run(out_softmax,
                                       feed_dict={input_x: 1.0 - np.array(img).reshape((-1,28, 28, 1)) / 255.0})

            print "img_out_softmax:", img_out_softmax
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print "label:", prediction_labels


pb_path = "/home/graydove/Graydove/NIMA/tensorflow_model/tensor_model.pb"
img = "/home/graydove/Graydove/NIMA/images/NIMA.jpg"
recognize(img, pb_path)