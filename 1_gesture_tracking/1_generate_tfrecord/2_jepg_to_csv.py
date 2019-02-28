# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:11:46 2018

@author: wuzhenguang
"""
from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
import cv2, os, time
import pandas as pd

width, height, channel = 480, 400, 3
root_path = './jepgs/'
image_path = os.path.join(root_path, 'train/')  # source images
save_path = os.path.join(root_path, 'train_result/') # result path


def main():
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # image to be read
    image_list = list()
    dirname = image_path.split(os.sep)[-2]
    if os.path.isdir(image_path):
        for file in os.listdir(image_path):
            if file.endswith('.jpg'):
                image_list.append(os.path.join(image_path, file))
        print("Record all video completely from " + image_path)
    else:
        raise FileNotFoundError(image_path + " doesn't exist.")

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, height, width, channel))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    start = time.time()
    value_list = []
    for num, image_name in enumerate(image_list, 1):
        filename = image_name.split(os.sep)[-1]  # extract the filename
        name, ext = os.path.splitext(filename)
        if num % 200 == 1:
            print("Process the image : " + name + " ... {}/{}".format(num, len(image_list)))
        image_raw = cv2.imread(image_name)
        # Feed image list through network
        image_raw = cv2.resize(image_raw, (width, height), cv2.INTER_AREA)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image_v = np.expand_dims((np.array(image_raw).astype('float') / 255.0) - 0.5, 0)

        _, _, scale_v, center_v, _, _ = sess.run(
            [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf],
            feed_dict={image_tf: image_v})

        # left top right bottom
        y_c, x_c = np.squeeze(center_v)
        half_side_len = 128.0 / scale_v  # the length of frame
        x, y = int(x_c - half_side_len*0.9), int(y_c - half_side_len)
        xmax, ymax = int(x_c + half_side_len*0.9), int(y_c + half_side_len)
        x, y, xmax, ymax = max(x, 0), max(y, 0), min(xmax, width), min(ymax, height)
        cv2.rectangle(image_raw, (x, y), (xmax, ymax), (77, 255, 9), 1, 1)
        res_img = name + '_v' + ext
        res_roi = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, res_img), cv2.resize(res_roi, (width//2, height//2), cv2.INTER_AREA))
        value = (filename, width, height, 'hand', x, y, xmax, ymax)
        value_list.append(value)
        
    endt = time.time()
    print("Handle all images completely and elapsed_time is {:.2f}s.".format(endt-start))
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(value_list, columns=column_name)
    xml_df.to_csv(os.path.join(root_path, dirname + '.csv'), index=None)
    print('Successfully converted image to csv. --- ' + dirname)


if __name__ == '__main__':
    main()
