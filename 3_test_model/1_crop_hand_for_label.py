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

width, height, channel = 768, 432, 3
input_size = 400
frames_path = '/home/zgwu/HandImages/long_video/frames/'
save_roi_path = '/home/zgwu/HandImages/long_video/test_rois/'


def which_label(file_name):
    labelsStr = ['Fist', 'Admire', 'Victory', 'Okay', 'None', 'Palm', 'Six']
    if file_name.startswith('F'):
        return labelsStr[0]
    elif file_name.startswith('A'):
        return labelsStr[1]
    elif file_name.startswith('V'):
        return labelsStr[2]
    elif file_name.startswith('O'):
        return labelsStr[3]
    elif file_name.startswith('N'):
        return labelsStr[4]
    elif file_name.startswith('P'):
        return labelsStr[5]
    elif file_name.startswith('S'):
        return labelsStr[6]
    else:
        return 'hand'


def main():
    if not os.path.isdir(save_roi_path):
        os.mkdir(save_roi_path)

    if os.path.isdir(frames_path):
        image_list = [os.path.join(frames_path, file) for file in os.listdir(frames_path) if file.endswith('.jpg')]
        print("Record {} image completely from  {} ".format(len(image_list), frames_path))
    else:
        raise NotADirectoryError(frames_path + " isn\'t a directory.")

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, input_size, input_size, channel))  # h, w, c
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

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    dist = (width - height)//2
    for num, image_name in enumerate(image_list, 1):
        filename = os.path.basename(image_name) #  extract the filename
        name, ext = os.path.splitext(filename)
        label = which_label(name)
        image_bgr = cv2.imread(image_name)
        # Feed image list through network
        # image_bgr = cv2.resize(image_raw, (width, height), cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if (num % 100 == 1):
            print("Process the image : " + name + " ... {}/{}".format(num, len(image_list)))

        x1, x2, y1, y2 = width//2 - dist, width//2 + dist, 0, height
        image_v = image_rgb[y1:y2, x1:x2]  # cenetr image for label
        image_v = cv2.resize(image_v, (input_size, input_size), cv2.INTER_AREA)
        image_v = np.expand_dims((np.array(image_v).astype('float') / 255.0) - 0.5, 0)

        _, _, scale_v, center_v, _, _ = sess.run(
            [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf],
            feed_dict={image_tf: image_v})

        # left top right bottom
        y_c, x_c = np.squeeze(center_v)
        half_side_len = 128.0 / scale_v  # the length of frame
        x, y = max(x1 + int(x_c - half_side_len), 0), max(y1 + int(y_c - half_side_len), 0)
        xmax, ymax = min(x1 + int(x_c + half_side_len), width), min(y1 + int(y_c + half_side_len), height)

        image_crop = image_bgr[y:ymax, x:xmax]
        res_path = os.path.join(save_roi_path, filename)
        cv2.imwrite(res_path, image_crop)

        value = (filename, width, height, label, x, y, xmax, ymax)
        value_list.append(value)

    xml_df = pd.DataFrame(value_list, columns=column_name)
    xml_df.to_csv(os.path.join(frames_path, 'test_images.csv'), index=None)
    endt = time.time()
    print("Handle all images completely and elapsed_time is {:.2f}s.".format(endt-start))


if __name__ == '__main__':
    main()
