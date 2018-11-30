# coding=utf-8
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import numpy as np
from utils import detector_utils as detector_utils
import os, cv2, time, datetime

video_path = 'videos\\'   # 检测视频放在该目录下
frame_path = 'frames\\'   # 按‘s’键截取图片，并保存在该目录下
labelsStr = ['Fist', 'Admire', 'Victory', 'Okay', 'None', 'Palm', 'Hex']
labelsDict = {'Fist': 0, 'Admire': 1, 'Victory': 2, 'Okay': 3, 'None': 4, 'Palm': 5, 'Hex': 6}
CKPT = 'weights\\model.ckpt-16000'  # 请选择最佳的ckpt文件
num_classes = len(labelsStr)
score_thresh = 0.1    # score_thresh of detecting hand
num_hands_detect = 2  # max number of hands we want to detect/track
pretty_width, pretty_height = 700, 400  # design how wide and high to show and detect
gap = 3


def from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, width, height, image_np):
    hands = list()
    flag = False
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            (left, right, top, bottom) = (int(boxes[i][1] * width), int(boxes[i][3] * width),
                                          int(boxes[i][0] * height), int(boxes[i][2] * height))
            xbias, ybias = (right - left) / 4, (bottom - top) / 5  # adjust the hand region to crop
            # factor = ybias / xbias
            (left, right, top, bottom) = (max(0, left - int(xbias)), min(right + int(xbias), width),
                                          max(0, top - int(1.3*ybias)), min(bottom + int(0.7*ybias), height))
            region = image_np[top:bottom, left:right]
            region = cv2.resize(region, (224, 224), cv2.INTER_AREA)
            flag = True
            rect = (left, right, top, bottom)
            hands.append((region, rect))
    return hands, flag


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

    out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=num_classes,
                                             depth_multiplier=1.0, is_training=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore variables that have been trained.
        saver.restore(sess, CKPT)

        # network
        detection_graph, sessd = detector_utils.load_inference_graph()  # ssd to detect hands

        for video_name in video_list:
            name = 'camera'
            if flag:
                filename = video_name.split(os.sep)[-1]
                name, ext = os.path.splitext(filename)
            start = time.time()
            cap = cv2.VideoCapture(video_name)
            # origin_fps = '%.2f' % cap.get(cv2.CAP_PROP_FPS)
            total_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            im_width, im_height = cap.get(3), cap.get(4)  # 1920,1080 | 1280 720 horizontal default
            factor = max(min(im_width / pretty_width, im_height / pretty_height), 1.0)  # 最佳缩放因子
            im_width, im_height = int(im_width / factor), int(im_height / factor)
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

                # resize to a small size to reduce caculation.
                image_np = cv2.resize(image_np, (im_width, im_height))

                all_frames += 1
                if gap > 1 and all_frames % gap != 1:  # skip gap-1 frames
                    continue
                num_frames += 1

                try:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                except:
                    print("Error converting to RGB")

                # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                # while scores contains the confidence for each of these boxes.
                # Hint: if len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)
                boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sessd)

                # draw bounding boxes on frame
                # detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

                # crop bounding boxes on frame
                hands, flag = from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)
                if not flag:
                    cv2.putText(image_np, "No result.", (10, 20), font, 0.5, (0, 255, 0), 1, 1)
                    cv2.imshow("Detect-hand", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                    continue

                key = cv2.waitKey(5) & 0xff  ## Use Esc key to close the program
                if key == 27:
                    break

                rank = 0  # the rank of hand
                for hand in hands:
                    rank += 1
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
                    # print("Result index = %d." % index + ' with ' + str(predictions))
                    (left, right, top, bottom) = rect
                    cv2.putText(image_np, labelsStr[index] + " : %%%d" % int(predictions[index]*100),
                                (left, top-8), font, 0.35, (0, 255, 255), 1, 1)
                    cv2.rectangle(image_np, (left, top), (right, bottom), (77, 255, 9), 1, 1)

                    if key == ord('s'):
                        region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                        cv2.imshow(labelsStr[index] + '-crop', region)
                        cv2.imwrite(frame_path + name + '_' + str(num_frames) + '_' + str(rank) + '.jpg', region)
                        cv2.waitKey(0)

                elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                fps = 'fps : %.2f ' % (num_frames / elapsed_time)
                cv2.putText(image_np, fps + info, (10, 20), font, 0.4, (0, 255, 0), 1, 1)  # current fps with size info
                # show frame
                cv2.imshow("Detect-hand", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            # Release & Destroy
            cap.release()
            cv2.destroyAllWindows()

            endr = time.time()
            print("Handle {} frames and elapsed_time is {:.2f}s.".format(total_len, (endr - start)))
        print("Bye.")


if __name__ == '__main__':
    test_model_from_video(video_path)
