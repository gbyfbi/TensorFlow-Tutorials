import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

# INCEPTION_LOG_DIR = '/tmp/inception_v3_log'
INCEPTION_LOG_DIR = '/tmp/vgg_16'

if not os.path.exists(INCEPTION_LOG_DIR):
    os.makedirs(INCEPTION_LOG_DIR)
with tf.Session() as sess:
    # model_filename = '/home/gao/Dropbox/Deeplearning/tensorflow/models/imagenet/inception-2015-12-05/classify_image_graph_def.pb'
    model_filename = '/home/gao/Data/flower/tf_data/slim/flower_fine_tune/checkpoints/vgg_16.ckpt'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter(INCEPTION_LOG_DIR, sess.graph)
    writer.close()
