# coding=utf-8
import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import numpy as np
from utils import detector_utils as detector_utils
import os, cv2, time, datetime, platform
import logging

# user settings
video_path = 'video\\'
frame_path = 'frame\\'
log_path = 'test_log\\'
score_thresh = 0.5    # score_thresh of detecting hand
num_hands_detect = 1  # max number of hands we want to detect/track
pretty_width, pretty_height = 700, 400  # design how wide and high to show and detect
gap = 3
b_show = False

# label settings
labelsStr = ['Fist', 'Admire', 'Victory', 'Okay', 'None', 'Palm', 'Six']
label = {'Fist': 0, 'Good': 1, 'V': 2, 'OK': 3, 'None': 4, 'Palm': 5, 'Six': 6}
labelsDict = {'Fist': 0, 'Admire': 1, 'Victory': 2, 'Okay': 3, 'None': 4, 'Palm': 5, 'Hex': 6}
CKPT = 'weights\\1203_s\\best_models_6000_0.9643.ckpt'  #待填写ckpt的路径
num_classes = len(labelsStr)
settings = '\n@@check_point_file: ' + CKPT + \
           '\n@@hand_score_thresh: ' + str(score_thresh) + \
           '\n@@max_num_hands_detect: ' + str(num_hands_detect) + \
           '\n@@' + str(platform.uname())


def from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, width, height, image_np):
    hands = list()
    flag = False
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            (left, right, top, bottom) = (int(boxes[i][1] * width), int(boxes[i][3] * width),
                                          int(boxes[i][0] * height), int(boxes[i][2] * height))
            xbias, ybias = (right - left) // 4, (bottom - top) // 5  # adjust the hand region to crop
            # factor = ybias / xbias
            (left, right, top, bottom) = (max(0, left - xbias), min(right + xbias, width),
                                          max(0, top - int(1.5*ybias)), min(bottom + int(0.3*ybias), height))
            w, h = right - left, bottom - top
            if w > h:
                top = max(top - (w - h) + ybias, 0)
            elif h > int(1.3 * w):
                top += (h-int(1.3*w))
            region = image_np[top:bottom, left:right]
            region = cv2.resize(region, (224, 224), cv2.INTER_AREA)
            flag = True
            rect = (left, right, top, bottom)
            hands.append((region, rect))
    return hands, flag


def read_video(video_path):
    video_list = list()
    flag = False
    if os.path.isdir(video_path):
        for file in os.listdir(video_path):
            if file.endswith(('.mp4', '.MP4', '.avi', '.wav')):
                video_list.append(os.path.join(video_path, file))
        print("Record all videos pathname completely from " + video_path)
        flag = True
    else:
        # video_list.append(0)
        raise NotADirectoryError(video_path + " doesn't exist.")
    return video_list, flag


def test_model_from_video(video_path, logger):
    video_list, isVideo = read_video(video_path)
    # placeholder holds an input tensor for classification
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3],
                                  name='input')
    out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=num_classes,
                                             depth_multiplier=1.0, is_training=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore variables that have been trained.
        saver.restore(sess, CKPT)
        print("Rsetore from " + CKPT)
        # network
        detection_graph, sessd = detector_utils.load_inference_graph()  # ssd to detect hands
        G = {'tot_frames':0, 'tot_num_hands':0, 'tot_acc':0, 'tot_detect_t':0.0, 'tot_classify_t':0.0}
        for num_video, video_name in enumerate(video_list, 1):
            name = 'camera'
            if isVideo:
                filename = os.path.basename(video_name)
                name, ext = os.path.splitext(filename)

            elapsed_time = -time.time()
            print("Processing the video : " + name + " ... {}/{}".format(num_video, len(video_list)))

            cap = cv2.VideoCapture(video_name)
            # origin_fps = '%.2f' % cap.get(cv2.CAP_PROP_FPS)
            total_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            im_width, im_height = cap.get(3), cap.get(4)  # 1920,1080 | 1280 720 horizontal default
            factor = max(min(im_width / pretty_width, im_height / pretty_height), 1.0)  # 最佳缩放因子
            im_width, im_height = int(im_width / factor), int(im_height / factor)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, im_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, im_height)
            info = 'width : ' + str(im_width) + ' height : ' + str(im_height)
            font = cv2.FONT_HERSHEY_DUPLEX  # cv2.FONT_HERSHEY_SIMPLEX
            
            D = {'all_frames':0, 'num_frames':0, 'acc':0, 'num_hands':0, 'tot_detect_t':0.0, 'tot_classify_t':0.0, 'r':[0, 0, 0, 0, 0, 0, 0]}
            acc_index = label[str(name.split('_')[0])]
            detect_t, classify_t = 0.0, 0.0
            while True:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                success, image_np = cap.read()
                if not success:  # read empty frame and break anyways
                    break
                # resize to a small size to reduce caculation.
                image_np = cv2.resize(image_np, (im_width, im_height))

                D['all_frames'] += 1
                if gap > 1 and D['all_frames'] % gap != 1:  # skip gap-1 frames
                    if not isVideo:
                        cv2.waitKey(10)
                    continue
                D['num_frames'] += 1

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

                start = time.time()  # set time
                
                # crop bounding boxes on frame
                hands, flag = from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

                if not flag:
                    # cv2.putText(image_np, "No result.", (10, 20), font, 0.5, (0, 255, 0), 1, 1)
                    elapsed_time += time.time()
                    fps = 'fps : %.2f ' % (D['num_frames'] / elapsed_time)
                    cv2.putText(image_np, fps + info, (10, 20), font, 0.4, (0, 255, 0), 1, 1)
                    if b_show:
                        cv2.imshow("Detect-hand", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                    continue

                key = cv2.waitKey(5) & 0xff  ## Use Esc key to close the program
                if key == 27:
                    break
                if key == ord('p'):
                    cv2.waitKey(0)

                endh = time.time()  # detect hand
                detect_t = endh-start
                D['tot_detect_t'] += detect_t

                for rank, hand in enumerate(hands):
                    D['num_hands'] += 1
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
                    index = int(np.argmax(predictions))
                    D['r'][index] += 1
                    if index == acc_index:
                        D['acc'] += 1
                    # print("Result index = %d." % index + ' with ' + str(predictions))
                    (left, right, top, bottom) = rect
                    cv2.putText(image_np, labelsStr[index] + " : %%%d" % int(predictions[index]*100),
                                (left, top-8), font, 0.35, (77, 255, 9), 1, 1)
                    cv2.rectangle(image_np, (left, top), (right, bottom), (77, 255, 9), 1, 1)

                    if key == ord('s'):
                        region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                        # cv2.imshow(labelsStr[index] + '-crop', region)
                        cv2.imwrite(frame_path + name + '_' + str(D['num_frames']) + '_' + str(rank) + '.jpg', region)
                        cv2.waitKey(0)
                        
                endr = time.time()
                classify_t = endr-endh
                D['tot_classify_t'] += classify_t
                
                elapsed_time += time.time()
                fps = 'fps : %.2f ' % (D['num_frames'] / elapsed_time)
                cv2.putText(image_np, fps + info, (10, 20), font, 0.4, (0, 255, 0), 1, 1)  # current fps with size info
                # show frame
                if b_show:
                    cv2.imshow("Detect-hand", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                
            # Release & Destroy
            cap.release()
            cv2.destroyAllWindows()

            print("Detect {} frames with {} accurate prediction ({:.2f})".format(D['num_hands'], D['acc'], D['acc']/D['num_hands']))
            result_log = '\n@@video_name: ' + name + ' - [{}/{}]'.format(num_video, len(video_list)) + \
                         '\n@@video_size: (width : {}, height: {})'.format(im_width, im_height) + \
                         '\n@@total_frame: ' + str(int(total_len)) + ' and test_frame: ' + str(D['num_frames']) + \
                         '\n@@num_hand_detect: {} - {}%'.format(D['num_hands'], int(100*D['num_hands']/D['num_frames'])) + \
                         '\n@@each_elapsed_time: (detect_hands: {:.4f}s, classify_hand: {:.4f}s)'.format(D['tot_detect_t']/D['num_frames'], D['tot_classify_t']/D['num_hands']) + \
                         '\n@@classify_result: Fist  Admire  Victory  Okay  None  Palm  Six' + \
                         '\n                   {: <6d}{: <8d}{: <9d}{: <6d}{: <6d}{: <6d}{}'.format(D['r'][0], D['r'][1], D['r'][2], D['r'][3], D['r'][4], D['r'][5], D['r'][6]) + \
                         '\n@@accuracy: {}/{} - {}%'.format(D['acc'], D['num_hands'], int(100*D['acc']/D['num_hands'])) + \
                         '\n' + '-'*100
            print(result_log)
            logger.info(result_log)
            G['tot_num_hands'] += D['num_hands']
            G['tot_frames'] += D['num_frames']
            G['tot_acc'] += D['acc']
            G['tot_detect_t'] += D['tot_detect_t']
            G['tot_classify_t'] += D['tot_classify_t']
        logger.info("Totally detect {} frames with {} hand-detection ({}%) -- each_elapsed_time {:.4f}s\n"
                    "Recognize {} hands sucessfully and accurate prediction is ({}%) -- each_elapsed_time {:.4f}s\n"
                    "Bye.".format(G['tot_frames'], G['tot_num_hands'], int(100*G['tot_num_hands']/G['tot_frames']), G['tot_detect_t']/G['tot_frames'], G['tot_acc'], int(100*G['tot_acc']/G['tot_num_hands']), G['tot_classify_t']/G['tot_num_hands']))


if __name__ == '__main__':
    dat = datetime.datetime.now()  # 获取当前时间以便定义日志格式
    logPathName = log_path + 'test_mobile_{}{}_{}{}.log'.format(dat.month, dat.day, dat.hour, dat.minute)
    logging.basicConfig(level=logging.INFO, filename=logPathName, datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(settings)
    test_model_from_video(video_path, logger)
