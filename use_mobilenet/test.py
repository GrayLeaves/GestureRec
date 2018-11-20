# coding=utf-8
from config import R
import tensorflow as tf
from slim.nets.mobilenet import mobilenet_v2
import numpy as np

# placeholder holds an input tensor for classification
input_images = tf.placeholder(dtype=tf.float32, shape=[None, R.resize_height, R.resize_width, R.depths], name='input')

out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=R.num_classes,
                                         depth_multiplier=0.75,
                                         is_training=False)
saver = tf.train.Saver()

img_raw_data = tf.gfile.FastGFile('./imag/*.jpg', 'rb').read()
img_data = tf.image.decode_jpeg(img_raw_data)
# img_data = tf.image.per_image_standardization(img_data)
img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
# elements are in [0,1)

resized_img = tf.image.resize_images(img_data, size=[R.resize_height, R.resize_width], method=0)

with tf.Session() as sess:
    # decode an image
    img = resized_img.eval()
    img.resize([1, R.resize_height, R.resize_width, R.depths])

    # restore variables that have been trained.
    saver.restore(sess, None) # 待填写ckpt的路径

    # input an image array and inference to get predictions and set normal format
    predictions = end_points['Predictions'].eval(feed_dict={input_images: img})
    predictions.resize([R.num_classes])
    np.set_printoptions(precision=4, suppress=True)
    print(predictions)