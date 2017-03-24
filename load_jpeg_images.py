# from https://gist.github.com/eerwitt/518b0c9564e500b4b50f and comment of batch
# Typical setup to include TensorFlow.
import tensorflow as tf
from glob import glob

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once("/home/gao/Downloads/flower_photos/daisy/*.jpg"))
file_name_list = glob("/home/gao/Downloads/flower_photos/daisy/*.jpg")
# filename_queue = tf.train.string_input_producer(["/home/gao/Downloads/flower_photos/daisy/2331133004_582772d58f_m.jpg"])
filename_queue = tf.train.string_input_producer(file_name_list, num_epochs=3)
# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)
# image_file_content = tf.read_file()

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image_orig = tf.image.decode_jpeg(image_file, channels=3)
image = tf.image.resize_images(image_orig, [224, 224])
image.set_shape((224, 224, 3))
batch_size = 50
num_preprocess_threads = 1
min_queue_examples = 256
images = tf.train.shuffle_batch(
    [image],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)
# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    # tf.initialize_all_variables().run()
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run(images)
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
