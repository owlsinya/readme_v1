import tensorflow as tf
import data.function as datafunc
import numpy as np

FILENAME = '/Users/sanglin/Documents/Competition_201809/0903testdata.csv'
RESTORE_CHK_POINT_PATH = 'pywavelet_model/0903train-9500'
df, datalist = datafunc.load_test_data(FILENAME,10)
print(datalist.shape)

#parameters
n_variables = 352
n_examples = 10

xs = tf.placeholder(tf.float32, [n_examples, n_variables], name="input")
#ys = tf.placeholder(tf.float32, [n_examples,1], name="output")

#model
#model
l1 = tf.layers.dense(xs,176, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
l2 = tf.layers.dense(l1,88, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
l3 = tf.layers.dense(l2,44, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
l4 = tf.layers.dense(l3,22, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
l5 = tf.layers.dense(l4,11, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
prediction = tf.layers.dense(l5, 1, activation=None)

#loss = tf.losses.mean_squared_error(prediction, ys) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

#saver load
saver = tf.train.Saver()


# para
with tf.Session() as sess:
    saver.restore(sess, RESTORE_CHK_POINT_PATH)
    result = sess.run(prediction, feed_dict={xs: ((datalist.values - datalist.values.min())/(datalist.values.max() - datalist.values.min()))})
    print()
    print('Prediction: ' ,result)
