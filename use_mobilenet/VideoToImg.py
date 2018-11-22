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
# @name : wzg
# @time : 2018-11-20
# @desc : extract each frame and crop the hand as dataset
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
import cv2, os, time, math

# 设置最后得到的图片高宽
height, width, channel = 320, 320, 3

video_path = './Videos/'               # source video path
result_root_path = './TargetImgPath/'  # result image path
# save pre-process pictures

def cropImgFromFrame(frame, area=None):
    row, col = int(height*1.5), int(width*1.5)  # input into hand3d (480, 480)
    yy, xx, _ = frame.shape
    # bias = 20
    # x,y,w,h -> crop[y:y+h, x:x+w]
    l, r = (xx - yy) // 2, (xx + yy) // 2
    xpos, ypos, ww, hh = l, 0, yy, yy
    region = frame[0:yy, l:l+yy]
    if area != None:
        x, y, w, h = area # face area
        xpos, ypos = max(l, int(x - 2.5 * w)), max(0, int(y - 0.2*h))
        ww = min(int(x + 2.5 * w), r) - xpos
        hh = min(int(y + 2.0 * h), yy) - ypos
    if ww > col and hh > row:    
        region = frame[ypos:ypos+hh, xpos:xpos+ww] # crop Image
    ry, rx, _= region.shape
    s = 1  # 缩放系数
    if rx > width and ry > height:
        s = max(math.ceil(rx // col), math.ceil(ry // row))
    resROI = cv2.resize(region, (rx // s, ry // s), cv2.INTER_AREA)
    return resROI

def preProcessSingleFrame(frame):
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    faceCascade = cv2.CascadeClassifier('haarcascade_facedetect_alt.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    flag = False
    res_list = list()
    if len(faces) != 0:
        flag = True
        num = 0
        for (x, y, w, h) in faces:
            num += 1
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) 
            result = cropImgFromFrame(frame, (x, y, w, h))
            res_list.append(result)
    else:
        result = cropImgFromFrame(frame)
        res_list.append(result)
        
    return res_list, flag


def main():
    # video to be read
    video_list = list()
    if os.path.isdir(video_path):
        for file in os.listdir(video_path):
            if file.endswith(('.mp4', '.avi', '.wav')):
                video_list.append(video_path + '/' + file)
        print("Record all video completely from " + video_path)
    else:
        raise FileNotFoundError(video_path + " doesn't exist.")

    if not os.path.isdir(result_root_path):
        os.mkdir(result_root_path)
        
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

    num = 0
    for video_name in video_list:
        num += 1
        file = video_name.split(os.sep)[-1]  # extract the filename
        name, ext = os.path.splitext(file)

        res_path = result_root_path + name + os.sep  # result path
        if not os.path.isdir(res_path):
            os.mkdir(res_path)

        print("Process the video : " + name + " ... {}/{}".format(num, len(video_list)))
        
        start = time.time()
        vidcap = cv2.VideoCapture(video_name)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # total_len = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        gap = fps // 5  # each frame between 0.2s
        startPos = 3
        times, video_len = 0, 0
        success = True
        image_list = list()
        while success:
            success, frame = vidcap.read()
            video_len += 1
            if video_len % gap == startPos: # 处理该帧
                image_list, flag = preProcessSingleFrame(frame) #当检测到多张脸则返回多帧
                times += 1 if flag else 0
                for image in image_list: # BGR -> RGB
                    # Feed image list through network
                    image_raw = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #scipy.misc.imread(img_path_name)
                    image_raw = scipy.misc.imresize(image_raw, (height, width))
                    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

                    hand_scoremap_v, image_crop_v, scale_v, center_v, \
                    keypoints_scoremap_v, keypoint_coord3d_v = sess.run(
                        [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf],
                        feed_dict={image_tf: image_v})

                    # hand_scoremap_v = np.squeeze(hand_scoremap_v)
                    image_crop_v = np.squeeze(image_crop_v)
                    # keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
                    # keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

                    # post processing
                    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
                    # coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
                    # coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

                    # visualize
                    plt.figure(1)
                    plt.imshow(image_crop_v)  # crop palm
                    plt.axis('off')
                    fig = plt.gcf()
                    fig.set_size_inches(1.0, 1.0)  # dpi = 300, output = 1.0*300 pixels
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    output_path = res_path + name + str(video_len // gap) + '_h.jpg'
                    fig.savefig(output_path, format='jpg', transparent=True, dpi=300, pad_inches=0)
                    plt.close()
                    
        endr = time.time()
        print("Handle {} frames with {} face-detection and elapsed_time is {:.2f}s."
              .format(video_len // gap, times, (endr - start)))
        # shutil.rmtree(res_path)

    print("Handle all videos completely.")


if __name__ == '__main__':
    main()
