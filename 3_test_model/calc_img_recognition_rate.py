# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.mobilenet import mobilenet_v2
from nets import inception_v3 as inception_v3
import numpy as np
import pandas as pd
from utils import detector_utils as detector_utils
import os, cv2, time, datetime, platform
from itertools import cycle
import logging
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn import metrics
#from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import interp

frame_path = 'crop_frames/'
log_path = 'test_log/'
score_thresh = 0.3  # score_thresh of detecting hand
num_hands_detect = 1  # max number of hands we want to detect/track
im_width, im_height = 768, 432  #768, 432
root = '/home/zgwu/HandImages'
labelsStr = ['Fist', 'Admire', 'Victory', 'Okay', 'None', 'Palm', 'Hex']
labelsDict = {'Fist': 0, 'Admire': 1, 'Victory': 2, 'Okay': 3, 'None': 4, 'Palm': 5, 'Hex': 6}
CKPT = 'weights/12 3_s/best_models_6000_0.9643.ckpt'  #mobilenet_v2
#CKPT = 'weights/mdh041807/model_at_18200_acc_0.984.ckpt' #inception_v3
num_classes = len(labelsStr)
settings = '\n@@check_point_file: ' + CKPT + \
           '\n@@hand_score_thresh: ' + str(score_thresh) + \
           '\n@@max_num_hands_detect: ' + str(num_hands_detect) + \
           '\n@@' + str(platform.uname())
ModelType = 'mobilenet' #'inception' #


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


def from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, width, height, iou_thresh):
    hands = list()
    detection = []
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            (left, right, top, bottom) = (int(boxes[i][1] * width), int(boxes[i][3] * width),
                                          int(boxes[i][0] * height), int(boxes[i][2] * height))
            w, h = right - left, bottom - top  # adjust the hand region to crop
            xc, yc = left + w // 2, top + h // 2
            len = min(int((w + h)*1.6), 240) // 4
            (left, right, top, bottom) = (max(0, xc - len), min(xc + len, width),
                                          max(0, yc - len), min(yc + len, height))
            detection.append([left, top, right, bottom, scores[i]])
    if not detection:
        return hands
    detect_array = np.array(detection)
    keep = nms(detect_array, iou_thresh)
    for item in detect_array[keep]:
        left, top, right, bottom, _ = item
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        rect = (left, top, right, bottom)
        hands.append(rect)
    return hands


def read_test_csv_from(csv_path):
    label_dict = dict()
    csv_data = pd.read_csv(csv_path)
    for i, row in csv_data.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        clazz = row['class']
        label_dict[row['filename']] = (x1, y1, x2, y2, clazz)
    return label_dict


def run_test(read_csv, logger):
    img_folder = '/home/zgwu/HandImages/long_video/test_frames/'
    save_folder = '/home/zgwu/HandImages/long_video/double_frames/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    label_dict = read_test_csv_from(read_csv)
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    # placeholder holds an input tensor for classification
    if ModelType == 'mobilenet':
        out, end_points = mobilenet_v2.mobilenet(input_tensor=input_images, num_classes=num_classes,
                                             depth_multiplier=1.0, is_training=False)
    else:
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=num_classes,
                                             is_training=False)
    detection_graph, sessd = detector_utils.load_inference_graph()  # ssd to detect hands
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, CKPT)
    CM = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]
    D = {'num_pictures': 0, 'acc': 0, 'num_hands': 0, 'detect_t': 0.0, 'classify_t': 0.0,
         'o': [0, 0, 0, 0, 0, 0, 0], 'd' : [0, 0, 0, 0, 0, 0, 0],
         'r': [0, 0, 0, 0, 0, 0, 0], 'tp' : [0, 0, 0, 0, 0, 0, 0]}
    tot_count = 0
    y_true, y_pred = [], []

    label_matrix = np.empty((0, num_classes), dtype=int)
    score_matrix = np.empty((0, num_classes), dtype=int)
    for num_img, (img_name, frame_label) in enumerate(label_dict.items()):
        tot_count += 1
        if tot_count % 100 == 1:
            print('Process {} --- {} / {}'.format(img_name, tot_count, len(label_dict)))
        l, t, r, b, clazz = frame_label
        acc_index = labelsDict[clazz]
        D['o'][acc_index] += 1                 # confusion matrix

        # filename = os.path.basename(img_name)
        name, ext = os.path.splitext(img_name)
        # print("Processing the image : " + name + " ... {}/{}".format(num_img+1, len(label_dict)))
        key = cv2.waitKey(5) & 0xff  ## Use Esc key to close the program
        if key == 27:
            break
        if key == ord('p'):
            cv2.waitKey(0)
        image_raw = cv2.imread(os.path.join(img_folder, img_name))
        image_np = cv2.resize(image_raw, (im_width, im_height))
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
        start = time.time()  # set time
        boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sessd)
        hands = from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, 0.5)
        endh = time.time()  # detect hand
        detect_t = endh - start
        D['detect_t'] += detect_t
        if not hands:
            continue
        else:
            D['num_pictures'] += 1
        D['d'][acc_index] += 1
        for rank, rect in enumerate(hands):
            D['num_hands'] += 1
            left, top, right, bottom = rect
            region = image_np[top:bottom, left:right]
            region = cv2.resize(region, (224, 224), cv2.INTER_AREA)
            # feed = np.expand_dims(region, axis=0)   # maybe it's a wrong format to feed
            img_data = tf.image.convert_image_dtype(np.array(region)[:, :, 0:3], dtype=tf.float32)  # RGB
            # elements are in [0,1)
            resized_img = tf.image.resize_images(img_data, size=[224, 224], method=0)
            # decode an image
            img = resized_img.eval(session=sess)
            img.resize([1, 224, 224, 3])
            # input an image array and inference to get predictions and set normal format
            predictions = end_points['Predictions'].eval(session=sess, feed_dict={input_images: img})

            label = np.zeros((1, num_classes), dtype=int)
            label[0, acc_index] = 1
            label_matrix = np.append(label_matrix, label, axis=0)
            score_matrix = np.append(score_matrix, predictions.reshape([1, num_classes]), axis=0)
            #print(label, predictions.reshape([1, num_classes]))

            predictions.resize([num_classes])
            np.set_printoptions(precision=4, suppress=True)
            index = int(np.argmax(predictions))
            y_true.append(acc_index)
            y_pred.append(index)
            D['r'][index] += 1
            msg = img_name + ' ' + clazz + ' ' + labelsStr[index]
            CM[acc_index][index] += 1

            if index == acc_index:
                D['acc'] += 1
                D['tp'][index] += 1

            logger.info(msg)
            if key == ord('s'):
                region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path + name + '_' + str(D['num_frames']) + '_' + str(rank) + '.jpg', region)
                cv2.waitKey(0)
        endr = time.time()
        classify_t = endr-endh
        D['classify_t'] += classify_t
    print("From {} pictures, we detect {} hands with {} accurate prediction ({:.2f})"
          .format(tot_count, D['num_hands'], D['acc'], D['acc'] / D['num_hands']))
    result_log = '\n@@images_count: {} and detect_count: {}'.format(tot_count, D['num_pictures']) + \
                 '\n@@image_size: (width : {}, height: {})'.format(im_width, im_height) + \
                 '\n@@num_hand_detect: {} - {}%'.format(D['num_hands'], int(100 * D['num_hands'] / tot_count)) + \
                 '\n@@each_elapsed_time: (detect_hands: {:.4f}s, classify_hand: {:.4f}s)'.format(
                     D['detect_t'] / tot_count, D['classify_t'] / D['num_hands']) + \
                 '\n@@classify_result: Fist  Admire  Victory  Okay  None  Palm  Six' + \
                 '\n                   {: <6d}{: <8d}{: <9d}{: <6d}{: <6d}{: <6d}{} -- origin classes' \
                 '\n                   {: <6d}{: <8d}{: <9d}{: <6d}{: <6d}{: <6d}{} -- detect classes' \
                 '\n                   {: <6d}{: <8d}{: <9d}{: <6d}{: <6d}{: <6d}{} -- recognize count' \
                 '\n                   {: <6d}{: <8d}{: <9d}{: <6d}{: <6d}{: <6d}{} -- true positive' \
                     .format(D['o'][0], D['o'][1], D['o'][2], D['o'][3],D['o'][4], D['o'][5], D['o'][6],
                     D['d'][0], D['d'][1], D['d'][2], D['d'][3],D['d'][4], D['d'][5], D['d'][6],
                     D['r'][0], D['r'][1], D['r'][2], D['r'][3], D['r'][4], D['r'][5], D['r'][6],
                     D['tp'][0], D['tp'][1], D['tp'][2], D['tp'][3], D['tp'][4], D['tp'][5], D['tp'][6]) + \
                 '\n@@accuracy: {}/{} - {}%'.format(D['acc'], D['num_hands'], int(100 * D['acc'] / D['num_hands'])) + \
                 '\n' + '-' * 100 + \
                 '\n' + str(CM)
    #print(result_log)
    logger.info(result_log)
    #print(classification_report(y_true, y_pred, target_names=labelsStr, digits=3))
    logger.info(str(classification_report(y_true, y_pred, target_names=labelsStr, digits=3)))

    print(label_matrix.shape, score_matrix.shape)
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(label_matrix.ravel(), score_matrix.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr["micro"], tpr["micro"], c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % roc_auc["micro"])
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'The ROC and AUC of MobileNet Classifier.', fontsize=17)
    plt.show()
    """
    fpr, tpr, thresholds = roc_curve(y_binary, y_scores)
    AUC = auc(fpr, tpr)
    print("fpr -- tpr")
    (size, ) = fpr.shape
    for i in range(size):
        print(fpr[i], tpr[i])
    print("AUC " + str(AUC))
    plt.plot(fpr, tpr)
    plt.title('ROC_curve' + '(AUC: ' + str(AUC) + ')')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    """


if __name__ == '__main__':
    which_csv = '/home/zgwu/HandImages/long_video/images2.csv'
    dat = datetime.datetime.now()  # 获取当前时间以便定义日志格式
    logPathName = log_path + 'test_hand_{}{}_{}{}.log'.format(dat.month, dat.day, dat.hour, dat.minute)
    logging.basicConfig(level=logging.INFO, filename=logPathName, datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(settings)
    run_test(which_csv, logger)
