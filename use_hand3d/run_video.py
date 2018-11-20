#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
import time
import shutil
from PIL import Image
import math

height, width, channel = 320, 320, 3
video_path = './videos/'
save_root = './images/' # save pre-process pictures

def cropImg(img_name, area = None):
    pic = Image.open(img_name)
    xx, yy = pic.size
    bias = 10
    size = ((xx-yy)//2-bias, 0, (xx+yy)//2+bias, yy)
    if area != None:
        x, y, w, h = area
        left = max((xx-yy)//2-bias, int(x-2.2*w))
        right = min(int(x+1.5*w), (xx+yy)//2+bias)
        up = max(0, y)
        down = min(int(y+2.5*h), yy)
        if left < right-width and up < down-height:
            size = (left, up, right, down)
    # print('region' + str(size))
    region = pic.crop(size)
    rx, ry = region.size
    s = 1  #
    if rx > width and ry > height:
        s = max(math.ceil(rx // width), math.ceil(ry // height))
    res = region.resize((rx // s, ry // s), Image.ANTIALIAS)
    return res
    
def preprocess(img_list):
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
        
    res_list = list()
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    faceCascade = cv2.CascadeClassifier('haarcascade_facedetect_alt.xml')
    
    for img_name in img_list:
        filename = img_name.split("/")[-1]
        name, ext = os.path.splitext(filename)
        img = cv2.imread(img_name)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

        if len(faces) != 0:
            num = 0
            for (x, y, w, h) in faces:
                num += 1
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                result = cropImg(img_name, (x, y, w, h))
                save_path = save_root + str(num) + '_' + filename
                result.save(save_path)
                res_list.append(save_path)
                # cv2.imwrite(save_root + 'v_' + filename, img)
                # print("Save " + filename + ' Ok with face detection.')
        else:
            save_path = save_root + '0_' + filename
            result = cropImg(img_name)
            result.save(save_path)
            res_list.append(save_path)
            # print("Save " + filename + ' Ok.')
            # print("Can't find any faces.")
    return res_list

def main():
    # video to be read
    video_list = list()
    if os.path.exists(video_path):
        for file in os.listdir(video_path):
           if file.endswith(('.mp4', '.avi', '.wav')):
               video_list.append(video_path + '/' + file)
    
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, height, width, channel))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)
    
    num = 0
    for video_name in video_list:
        num += 1
        file = video_name.split("/")[-1]  # extract the filename
        name, ext = os.path.splitext(file)
        dir_path = video_path + name + "/" # frame-path
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        res_path = video_path + name + "_res/" # result path
        if not os.path.isdir(res_path):
            os.mkdir(res_path)
                
        print("Process the video : " + name + " ... {}/{}".format(num, len(video_list)))
        print("Start to set time.")
        start = time.time()
        
        vidcap = cv2.VideoCapture(video_name)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        gap = fps // 5  # each frame between 0.2s
        startPos = 3
        video_len = 0
        cnt = 0
        success = True
        img_list = list()
        while success:
            success, image = vidcap.read()
            video_len += 1
            if video_len % gap == startPos:
                frame_path = dir_path + name + "_%d.jpg" % (video_len // gap)
                cv2.imwrite(frame_path, image)  # save frame as JPEG file
                img_list.append(frame_path)

        endr = time.time()
        print("Save all frame to jpeg and elapsed_time is {:.2f}s.".format(endr-start))

        # read images and pre-process
        image_list = preprocess(img_list)
        print("Get %d files." %len(image_list))
        
        
        # Feed image list through network
        for img_name in image_list:
            imgName = img_name.split("/")[-1]  # extract the filename
            image_raw = scipy.misc.imread(img_name)
            image_raw = scipy.misc.imresize(image_raw, (height, width))
            image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

            hand_scoremap_v, image_crop_v, scale_v, center_v, \
            keypoints_scoremap_v, keypoint_coord3d_v = sess.run(
                    [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf],
                    feed_dict={image_tf: image_v})

            hand_scoremap_v = np.squeeze(hand_scoremap_v)
            image_crop_v = np.squeeze(image_crop_v)
            keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
            keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

            # post processing
            image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
            coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
            coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

            # visualize
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            # ax3 = fig.add_subplot(223)
            # ax4 = fig.add_subplot(224, projection='3d')
            ax1.imshow(image_raw)
            plot_hand(coord_hw, ax1)
            ax2.imshow(image_crop_v)
            plot_hand(coord_hw_crop, ax2)
            # ax3.imshow(np.argmax(hand_scoremap_v, 2))
            # plot_hand_3d(keypoint_coord3d_v, ax4)
            # ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            # ax4.set_xlim([-3, 3])
            # ax4.set_ylim([-3, 1])
            # ax4.set_zlim([-3, 3])
            # plt.show()

            plt.savefig(res_path + imgName)
            plt.close(fig)
            cnt += 1
            if (cnt == video_len):
                break
        ends = time.time()
        print("Saved all result picture and elapsed_time is {:.2f}s.".format(ends-endr))

        '''
        fps = 24
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(video_path + name + '_res.avi', fourcc, fps, (width, height))
        print("Let's view the video.")
        for i in range(video_len):
            frame = cv2.imread(res_path + str(i) + '.png')
            videoWriter.write(frame)
        videoWriter.release()
        endv = time.time()
        print("Save it as video and elapsed_time is {:.2f}s".format(endv-ends))
        '''
        # shutil.rmtree(dir_path)
        # shutil.rmtree(res_path)

    print("Handle all videos completely.")
    
if __name__ == '__main__':
    main()
    