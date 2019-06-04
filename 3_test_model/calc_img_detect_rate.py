# coding=utf-8
import numpy as np
import pandas as pd
from utils import detector_utils as detector_utils
import os, cv2, time
from itertools import groupby

frame_path = 'crop_frames/'
score_thresh = 0.0  # score_thresh of detecting hand
num_hands_detect = 20  # max number of hands we want to detect/track
im_width, im_height = 768, 432
root = '/home/zgwu/HandImages'
labelsStr = ['Fist', 'Admire', 'Victory', 'Okay', 'None', 'Palm', 'Hex']

def calcIoU(rect1, rect2):
    (x1, y1, x2, y2) = rect1
    (x3, y3, x4, y4) = rect2
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x4 - x3, y4 - y3
    iou_w = max(min(x2, x4) - max(x1, x3) + 1, 0)
    iou_h = max(min(y2, y4) - max(y1, y3) + 1, 0)
    intersection = iou_w * iou_h
    union = w1*h1 + w2*h2 - intersection
    return intersection / union


def center_mean(rect1, rect2):
    l, t, r, b = rect1
    cx, cy = (l + r) // 2, (t + b) // 2
    x1, y1, x2, y2 = rect2
    xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
    len = (r - l + x2 - x1 + b - t + y2 - y1) // 8
    return (cx-len, cy-len, cx+len, cy+len), (xc-len, yc-len, xc+len, yc+len)


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


def remove_item_from(src, remove_path, dst):
    assert os.path.isdir(remove_path), '{} must be dir'.format(remove_path)
    remove_set = set()
    for file in os.listdir(remove_path):
        if file.endswith('.jpg'):
            filename = os.path.basename(file)
            remove_set.add(filename)

    df = pd.read_csv(src, iterator=True)
    loop = True
    chunk_size = 100
    select_chunk = []
    while loop:
        try:
            chunk = df.get_chunk(chunk_size)
            select_chunk.append( chunk[ chunk['filename'].map(lambda item : not(item in remove_set)) ] )
        except StopIteration:
            loop = False
    select_row = pd.concat(select_chunk, ignore_index = True)
    select_row.to_csv(dst, index=False)


def read_hand_csv_from(csv_path, nbins):
    ious, scores = [], []
    iou_count, score_count = [], []
    csv_data = pd.read_csv(csv_path)
    for i, row in csv_data.iterrows():
        ious.append(float(row['iou']))
        scores.append(float(row['score']))

    for k, g in groupby(sorted(ious), key=lambda x : int(x*nbins)):
        iou_count.append(len(list(g)))
    for k, g in groupby(sorted(scores), key=lambda x : int(x*nbins)):
        score_count.append(len(list(g)))
    return iou_count, score_count


def read_hand_csv_and_adjust_label(frm_path, to_path):
    csv_data = pd.read_csv(frm_path)
    value_list = []
    for i, row in csv_data.iterrows():
        filename = str(row['filename'])
        if filename.startswith(('07_1_', '7_1_')):
            clazz = labelsStr[1]
        elif filename.startswith(('07_2_', '7_2_')):
            clazz = labelsStr[2]
        elif filename.startswith(('07_6_', '7_6_')):
            clazz = labelsStr[6]
        elif filename.startswith('7_3_'):
            clazz = labelsStr[3]
        elif filename.startswith('7_4_'):
            clazz = labelsStr[0]
        elif filename.startswith('7_5_'):
            clazz = labelsStr[5]
        value_list.append((filename, 768, 432, clazz, row['xmin'], row['ymin'], row['xmax'], row['ymax']))
    column_name = ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
    csv_df = pd.DataFrame(value_list, columns=column_name)
    csv_df.to_csv(to_path, index=None)
    print('Successfully record all item to csv. --- ' + to_path)


def read_test_csv_from(csv_path):
    label_dict = dict()
    csv_data = pd.read_csv(csv_path)
    for i, row in csv_data.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        label_dict[row['filename']] = (x1, y1, x2, y2)
    return label_dict


def record_tracker(read_csv, detect_csv):
    img_folder = '/home/zgwu/HandImages/long_video/test_frames/'
    save_folder = '/home/zgwu/HandImages/long_video/double_frames/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    label_dict = read_test_csv_from(read_csv)
    detection_graph, sessd = detector_utils.load_inference_graph()  # ssd to detect hands
    value_list = []
    start = time.time()
    all_factor = 0.0
    count = 0
    pass_count = 0
    for num_img, (img_name, frame) in enumerate(label_dict.items()):
        # filename = os.path.basename(img_name)
        name, ext = os.path.splitext(img_name)
        # print("Processing the image : " + name + " ... {}/{}".format(num_img+1, len(label_dict)))
        image_raw = cv2.imread(os.path.join(img_folder, img_name))
        image_np = cv2.resize(image_raw, (im_width, im_height))
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")
        boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sessd)
        hands = from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, 0.5)
        l, t, r, b = frame
        if not hands:
            value_list.append((img_name, 'hand', l, t, r, b, -1, -1, -1, -1, 0.0, 0.0))
            continue
        top = 0
        max_iou = 0.0
        for rank, rect in enumerate(hands):
            iou = calcIoU(rect, frame)
            if iou > max_iou:
                top = rank
                max_iou = iou
        if max_iou > 0.5:
            pass_count += 1
        x1, y1, x2, y2 = hands[top]
        cv2.rectangle(image_raw, (l, t), (r, b), (113,179,60), 1, 1)
        cv2.rectangle(image_raw, (x1, y1), (x2, y2), (255,144,30), 1, 1)
        cv2.imwrite(os.path.join(save_folder, img_name), image_raw)   # save double frame result
        value_list.append((img_name, 'hand', l, t, r, b, x1, y1, x2, y2, max_iou, scores[top]))
        all_factor += (r - l) / (x2 - x1)
        count += 1
    print("factor = {}, pass_count = {}".format(all_factor / count, pass_count))  # if mean factor is close to 1.0 that the lenght of both frame is fine
    endt = time.time()
    print("Handle {} images and elapsed_time is {:.2f}s.".format(len(label_dict), (endt - start)))
    column_name = ['filename', 'class', 'left', 'top', 'right', 'bottom', 'xmin', 'ymin', 'xmax', 'ymax', 'iou', 'score']
    csv_df = pd.DataFrame(value_list, columns=column_name)
    csv_df.to_csv(detect_csv, index=None)
    print('Successfully record all item to csv. --- ' + detect_csv)


if __name__ == '__main__':
    from_csv = '/home/zgwu/HandImages/long_video/test_frames/test_images.csv'
    to_csv = '/home/zgwu/HandImages/long_video/test_frames/images.csv'
    remove = '/home/zgwu/HandImages/long_video/remove/'
    # remove_item_from(from_csv, remove, to_csv)
    dst_csv = '/home/zgwu/HandImages/long_video/hands.csv'
    sve_csv = '/home/zgwu/HandImages/long_video/images2.csv'
    #record_tracker(to_csv, dst_csv)
    #iou_count, score_count = read_hand_csv_from(dst_csv, 50)
    #iou_sum, score_sum = np.cumsum(iou_count), np.cumsum(score_count)
    #print(iou_count, '\n', iou_sum)
    #print(score_count, '\n', score_sum)
    #for i in range(47):
    #   print(iou_count[i], iou_sum[i], score_count[i], score_sum[i], sep=' ')
    read_hand_csv_and_adjust_label(to_csv, sve_csv)
