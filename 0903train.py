import tensorflow as tf
import data.function as datafunc
import time
import numpy as np
"""
#data loading
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'quality': tf.FixedLenFeature([], tf.float32),
                                           'data_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    data = tf.decode_raw(features['data_raw'], tf.uint8)
    data = tf.cast(data, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['quality'], tf.float32)
    return data, label
data, label = read_and_decode("train.tfrecords")
print(data.get_shape().as_list())
print(label.get_shape())
"""

#data loading
FILENAME = '/Users/sanglin/Documents/Competition_201809/0903traindata.csv'
df, quality, datalist = datafunc.load_data(FILENAME,40)
quality = np.expand_dims(quality,axis=1)

def next_batch(num, data, labels):
    '''
    Return a total of 'num' random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#parameters
num_epochs = 10000
hidden_layers = 1
n_variables = 352
learning_rate = 0.0001
batch_size = 16

#placeholder initialize
xs = tf.placeholder(tf.float32, [batch_size, n_variables], name="input")
ys = tf.placeholder(tf.float32, [batch_size,1], name="output")

#model
l1 = tf.layers.dense(xs,176, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
l2 = tf.layers.dense(l1,88, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
l3 = tf.layers.dense(l2,44, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
l4 = tf.layers.dense(l3,22, activation=tf.tanh,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
l5 = tf.layers.dense(l4,11, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
prediction = tf.layers.dense(l5, 1, activation=None)

#loss
loss = tf.losses.mean_squared_error(prediction, ys) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

#optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#summary
summary_loss = tf.summary.scalar('loss', loss)
summary_l2 = tf.summary.histogram(l2.name,l2)
summary_all = tf.summary.merge_all()

#variables initialize
init = tf.global_variables_initializer()

#saver initialize
saver = tf.train.Saver()

#main function
with tf.Session() as sess:
    st = time.time()
    write = tf.summary.FileWriter('logs/', sess.graph)
    #corrdinator & queue runners
    sess.run(init)
    for i in range(num_epochs):
        x_batch, y_batch = next_batch(batch_size, datalist.values, quality)
        x_batch = (x_batch - x_batch.min())/(x_batch.max() - x_batch.min())
        _, to_summary = sess.run([train_op,summary_all], feed_dict={xs: x_batch , ys: y_batch})
        write.add_summary(to_summary, global_step=i)
        if i % 500 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_batch, ys: y_batch}))
            #save checkpoint
            saver.save(sess,'pywavelet_model/0903train', global_step=i)
