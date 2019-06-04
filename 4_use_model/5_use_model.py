# coding=utf-8
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import tensorflow.contrib.slim as slim
from nets import inception_v3 as inception_v3
from nets.inception_resnet_v2 import *
import numpy as np
from utils import detector_utils as detector_utils
import os, cv2, time, datetime

video_path = 'test_video/'
frame_path = 'frames/'
region_path = 'regions/'
labelsStr = ['Fist', 'Thumbs', 'Scissors', 'Okay', 'Others', 'Palm', 'Shaka']
labelsDict = {'Fist': 0, 'Admire': 1, 'Victory': 2, 'Okay': 3, 'None': 4, 'Palm': 5, 'Hex': 6}
ModelType =  'mobilenet' # 'inception' # 'resnet'
#mobilenet
CKPT = 'weights/1203_s/best_models_6000_0.9643.ckpt'
#CKPT = 'weights/mdh042922_mobilenet/model_at_11840_acc_0.956.ckpt'
#inception
#CKPT = 'weights/mdh041807/model_at_18200_acc_0.984.ckpt'
#CKPT = 'weights/mdh042921_inception/model_at_12800_acc_0.966.ckpt'
#resnet
#CKPT = 'weights/mdh051423_resnet/model_at_15360_acc_0.915.ckpt'
num_classes = len(labelsStr)
IoUthresh = 0.3
score_thresh = 0.0    # score_thresh of detecting hand
num_hands_detect = 2  # max number of hands we want to detect
pretty_width, pretty_height = 768, 432 #768, 432  # design how wide and high to show and detect
gap = 10
bias = 5
fontsize = 0.6
thickness = 1
color = (50, 205, 50)
scale = 2

#fontsize /= scale

def fliter(index, score):
    return True if score < 0.0 or index == 4 else False
    
def nms(detection, iou_thresh):
    # NMS广泛应用于目标检测算法中。其目的是为了消除多余的候选框，找到最佳的物体检测位置，这里使用多框检测并提取最大值作为手势检测结果
    # detection = np.array([[x1,y1,x2,y2,score],[...]])
    """Pure Python NMS baseline."""
    x1 = detection[:, 0]
    y1 = detection[:, 1]
    x2 = detection[:, 2]
    y2 = detection[:, 3]
    scores = detection[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个候选框的面积
    order = scores.argsort()[::-1]  # 按score降序,算法得到的score
    keep = []
    while order.size > 0:
        top = order[0]
        keep.append(top)
        xx1 = np.maximum(x1[top], x1[order[1:]])
        yy1 = np.maximum(y1[top], y1[order[1:]])
        xx2 = np.minimum(x2[top], x2[order[1:]])
        yy2 = np.minimum(y2[top], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[top] + areas[order[1:]] - inter)  # IOU
        inds = np.where(ovr <= iou_thresh)[0]  # 找到重叠度不高于阈值的矩形框索引
        order = order[inds + 1]  # 更新序列
    return keep  # detection[keep]

    
def from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, width, height, iou_thresh,image_np):
    hands = list()
    detection = []
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            (left, right, top, bottom) = (int(boxes[i][1] * width), int(boxes[i][3] * width),
                                          int(boxes[i][0] * height), int(boxes[i][2] * height))
            w, h = right - left, bottom - top  # adjust the hand region to crop
            xc, yc = left + w // 2, top + h // 2
            len = min(int(w + h)*1.3, int(w + h)*1.4 + 8) // 4
            (left, right, top, bottom) = (max(0, xc - len), min(xc + len, width),
                                          max(0, yc - len - bias), min(yc + len - bias, height))
            detection.append([left, top, right, bottom, scores[i]])
    if not detection:
        return hands
    detect_array = np.array(detection)   
    keep = nms(detect_array, iou_thresh)
    for item in detect_array[keep]:
        left, top, right, bottom, _ = item
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        region = image_np[top:bottom, left:right]
        region = cv2.resize(region, (224, 224), cv2.INTER_AREA)
        rect = (left, right, top, bottom)
        hands.append((region, rect))
        
    return hands


def read_video(video_path):
    video_list = list()
    flag = False
    if os.path.exists(video_path):
        for file in os.listdir(video_path):
            if file.endswith(('.mp4', '.MP4', '.avi', '.wav')):
                video_list.append(video_path + file)
        print("Record all videos pathname completely from " + video_path)
        flag = True
    else:
        video_list.append(0)
        # raise NotADirectoryError(video_path + " doesn't exist.")
    return video_list, flag


def test_model_from_video(video_path):
    video_list, flag = read_video(video_path)

    # placeholder holds an input tensor for classification
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3],
                                  name='input')

    # placeholder holds an input tensor for classification
    if ModelType == 'mobilenet':
        out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=num_classes,
                                             depth_multiplier=1.0, is_training=False)
    elif ModelType == 'inception':
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=num_classes,
                                             is_training=False)
    else:
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            out, end_points = inception_resnet_v2(inputs=input_images, num_classes=num_classes,
                                             is_training=False)
    saver = tf.train.Saver()
   
    with tf.Session() as sess:  ##1
        # restore variables that have been trained.
        saver.restore(sess, CKPT) ##2

        # network
        detection_graph, sessd = detector_utils.load_inference_graph()  # ssd to detect hands
        
        for num_video, video_name in enumerate(video_list, 1):
            name = 'camera'
            if flag:
                filename = os.path.basename(video_name)
                name, ext = os.path.splitext(filename)

            print("Processing the video : " + name + " ... {}/{}".format(num_video, len(video_list)))
            start = time.time()
            cap = cv2.VideoCapture(video_name)
            # origin_fps = '%.2f' % cap.get(cv2.CAP_PROP_FPS)
            total_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            raw_width, raw_height = cap.get(3), cap.get(4)  # 1920,1080 | 1280 720 horizontal default
            factor = max(min(raw_width / pretty_width, raw_height / pretty_height), 1.0)  # 最佳缩放因子
            im_width, im_height = int(raw_width / factor), int(raw_height / factor)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, im_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, im_height)
            info = 'width : ' + str(im_width) + ' height : ' + str(im_height)

            all_frames, num_frames = 0, 0
            font = cv2.FONT_HERSHEY_DUPLEX  # cv2.FONT_HERSHEY_SIMPLEX
            start_time = datetime.datetime.now()

            while True:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                success, image_np = cap.read()
                if not success:  # read empty frame and break anyways
                    break

                key = cv2.waitKey(2) & 0xff  ## Use Esc key to close the program
                if key == 27:
                    break
                if key == ord('p'):
                    cv2.waitKey(0)
                    
                # resize to a small size to reduce caculation.
                image_np = cv2.resize(image_np, (int(raw_width/scale), int(raw_height/scale)))

                all_frames += 1
                if all_frames < 10:
                    continue
                if gap > 1 and all_frames % gap != 1:  # skip gap-1 frames
                    continue
                num_frames += 1

                try:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) #不要改
                except:
                    print("Error converting to RGB")

                # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                # while scores contains the confidence for each of these boxes.
                # Hint: if len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)
                boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sessd)

                # draw bounding boxes on frame
                # detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)
                
                # crop bounding boxes on frame
                hands = from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, int(raw_width/scale), int(raw_height/scale), IoUthresh, image_np)
                if not hands:
                    # cv2.putText(image_np, "No result.", (10, 20), font, 0.5, (0, 255, 0), 1, 1)
                    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                    fps = 'fps : %.2f ' % (num_frames / elapsed_time)
                    cv2.putText(image_np, fps + info, (10, 20), font, fontsize, color, thickness, 1)
                    image_show = cv2.resize(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), (im_width, im_height))
                    cv2.imshow("Detect-hand", image_show)
                    continue
                
                for rank, hand in enumerate(hands, 1):
                    (region, rect) = hand
                    # feed = np.expand_dims(region, axis=0)   # maybe it's a wrong format to feed
                    img_data = tf.image.convert_image_dtype(np.array(region)[:, :, 0:3], dtype=tf.float32)  # RGB
                    # elements are in [0,1)
                    resized_img = tf.image.resize_images(img_data, size=[224, 224], method=0)
                    # decode an image
                    img = resized_img.eval(session=sess)
                    img.resize([1, 224, 224, 3])
                    # input an image array and inference to get predictions and set normal format
                    predictions = end_points['Predictions'].eval(session=sess, feed_dict={input_images: img})
                    predictions.resize([num_classes])
                    np.set_printoptions(precision=4, suppress=True)
                    index = np.argmax(predictions)
                    if fliter(index, predictions[index]):
                        continue
                    # print("Result index = %d." % index + ' with ' + str(predictions))
                    (left, right, top, bottom) = rect
                    cv2.putText(image_np, labelsStr[index],# + " : %d%%" % int(predictions[index]*100),
                                (left, top-8), font, fontsize, color, thickness, 1)
                    cv2.rectangle(image_np, (left, top), (right, bottom), color, thickness, 1)
                    
                    if key == ord('s'):
                        region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                        cv2.imshow(labelsStr[index] + '-crop', region)
                        cv2.imwrite(region_path + name + '_' + str(num_frames) + '_' + str(rank) + '.jpg', region)
                        cv2.waitKey(0)

                elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                fps = 'fps : %.2f ' % (num_frames / elapsed_time)
                cv2.putText(image_np, fps + info, (10, 20), font, fontsize, color, thickness, 1)  # current fps with size info
                # show frame
                image_show = cv2.resize(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), (im_width, im_height))
                cv2.imshow("Detect-hand", image_show)
                # cv2.imwrite(frame_path + name + '_' + str(num_frames) + '.jpg', image_show)

            # Release & Destroy
            cap.release()
            cv2.destroyAllWindows()

            endr = time.time()
            print("Handle {} frames and elapsed_time is {:.2f}s.".format(total_len, (endr - start)))
        print("Bye.")


if __name__ == '__main__':
    test_model_from_video(video_path)
