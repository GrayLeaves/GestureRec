# -*- coding: utf-8 -*-

from utils import detector_utils as detector_utils
import cv2, os, time

# settings
video_path = './Videos/'               # source video path
result_root_path = './TargetImgPath/'  # result image path
score_thresh = 0.9  # scores
gap = 3
width, height = 640, 360  # resize to process or show
save_w, save_h = 320, 320 

def from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, width, height, image_np, save_path_name):
    isSave = False
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            (left, right, top, bottom) = (int(boxes[i][1] * width), int(boxes[i][3] * width),
                                          int(boxes[i][0] * height), int(boxes[i][2] * height))
            xbias, ybias = (right - left) // 4, (bottom - top) // 5  # adjust the hand region to crop
            # factor = ybias / xbias
            (left, right, top, bottom) = (max(0, left - xbias), min(right + xbias, width),
                                          max(0, top - ybias), min(bottom + ybias, height))
            region = image_np[top:bottom, left:right]
            region = cv2.resize(region, (save_w, save_h), cv2.INTER_AREA)
            region = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
            isSave = cv2.imwrite(save_path_name, region)

    return isSave


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

    # network
    detection_graph, sess = detector_utils.load_inference_graph()

    num_hands_detect = 1  # max number of hands we want to detect/track

    num_video = 0
    for video_name in video_list:
        num_video += 1
        file = video_name.split(os.sep)[-1]  # extract the filename
        name, ext = os.path.splitext(file)

        res_path = result_root_path + name + os.sep  # result path
        if not os.path.isdir(res_path):
            os.mkdir(res_path)

        print("Processing the video : " + name + " ... {}/{}".format(num_video, len(video_list)), end=' ')

        start = time.time()
        cap = cv2.VideoCapture(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('Fps - %.2f.' %fps)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        total_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # im_width, im_height = (cap.get(3), cap.get(4))  # 1920,1080
        # print("The pixel of video/camera is " + str((im_width, im_height)))
        # cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
        num_frames, num_hand = 0, 0
        while True:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            success, image_np = cap.read()
            if not success:  # read empty frame
                break
            num_frames += 1
            if num_frames % gap != 1:  # skip gap-1 frames
                continue
            # factor = max(im_width // width, im_height // height)
            image_np = cv2.resize(image_np, (width, height), cv2.INTER_AREA)  # resize to detect

            # image_np = cv2.flip(image_np, 1)
            try:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score  threshold)

            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)

            # draw bounding boxes on frame
            # detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

            # crop bounding boxes on frame and save as jepg
            save_path_name = res_path + name + str(num_frames) + '_s.jpg'
            flag = from_image_crop_boxes(num_hands_detect, score_thresh, scores, boxes, width, height, image_np,
                                         save_path_name)
            if flag:
                num_hand += 1

        endr = time.time()
        print("Handle {} frames with {} hand-detection and elapsed_time is {:.2f}s."
              .format(total_len, num_hand, (endr - start)))

    print("Handle all videos completely.")


if __name__ == '__main__':
    main()
