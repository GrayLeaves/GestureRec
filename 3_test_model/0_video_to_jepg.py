# coding=utf-8

import cv2
import os
import time

# resize_hw = 432  # 5:4
# bias = 54
width, height = 768, 432
tot_video_path = '/home/zgwu/HandImages/long_video/'
save_jepgs_path = os.path.join(tot_video_path, 'frames/')
gap = 5


def main():
    beginning = time.time()
    # video to be read
    video_list = list()
    # images to be shown
    if os.path.exists(tot_video_path):
        for file in os.listdir(tot_video_path):
            if file.endswith(('.mp4', '.avi', '.wav')):
                video_list.append(tot_video_path + file)
        print("Record all video-path completely from " + tot_video_path)
    else:
        raise NotADirectoryError(tot_video_path + " doesn't exist.")

    if not os.path.isdir(save_jepgs_path):
        os.mkdir(save_jepgs_path)
        
    for num, video_name in enumerate(video_list, 1):
        file = os.path.basename(video_name)  # extract the filename
        name, ext = os.path.splitext(file)  # get rid of ext
        print("Process the video : " + name + " ... {}/{}".format(num, len(video_list)))
        
        start = time.time()
        cap = cv2.VideoCapture(video_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("For {}, fps => {} and gap => {}".format(file, fps, gap))
        startPos = 2
        video_len = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            video_len += 1

            if gap > startPos and video_len % gap == startPos:
                '''
                yy, xx, _ = frame.shape
                left, right = (xx - yy) // 2 - bias, (xx + yy) // 2 + bias
                region = frame[0:yy, left:right]
                if yy < resize_hw:
                    print("Skip the file :" + name)
                    continue
                '''
                region = cv2.resize(frame, (width, height), cv2.INTER_AREA)
                frame_path = os.path.join(save_jepgs_path, name + "_%d.jpg" % video_len)
                cv2.imwrite(frame_path, region)  # save frame as JPEG file
        endr = time.time()
        print("Save {} frame to image(.jpg) and elapsed_time is {:.2f}s.".format(video_len // gap, endr - start))
    print("Handle all videos completely.")
    ending = time.time()
    print("Total elapsed_time is {:.2f}s".format(ending-beginning))


if __name__ == '__main__':
    main()
