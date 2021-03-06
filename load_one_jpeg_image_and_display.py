#from http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
import tensorflow as tf
import numpy as np
from PIL import Image
# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("/home/gao/Downloads/flower_photos/daisy/*.jpg")) #  list of files to read
filename_queue = tf.train.string_input_producer(["/home/gao/Downloads/flower_photos/daisy/2331133004_582772d58f_m.jpg"]) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

# my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1): #length of your filename list
    image = my_img.eval() #here is your image Tensor :)

  print(image.shape)
  Image.fromarray(np.asarray(image)).show()

  coord.request_stop()
  coord.join(threads)