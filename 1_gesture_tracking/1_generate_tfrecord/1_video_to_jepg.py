# coding=utf-8

import cv2
import os
import time

resize_hw = 400
video_path = './video/'
save_path = './jepgs/'
NUM_PIC = 250

def main():
    print("It may take 1600s to complete (when resize_hw = 300) and"
          " Please check your parameters before you run it.")
    beginning = time.time()
    # video to be read
    video_list = list()
    # images to be shown
    if os.path.exists(video_path):
        for file in os.listdir(video_path):
            if file.endswith(('.mp4', '.avi', '.wav')):
                video_list.append(video_path + file)
        print("Record all video-path completely from " + video_path)
    else:
        raise NotADirectoryError(video_path + " doesn't exist.")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    for num, video_name in enumerate(video_list, 1):
        file = os.path.basename(video_name)  # extract the filename
        name, ext = os.path.splitext(file)  # get rid of ext
        dir_path = save_path + name + "/"  # frame save_path
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        print("Process the video : " + name + " ... {}/{}".format(num, len(video_list)))
        
        start = time.time()
        cap = cv2.VideoCapture(video_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        gap = int(total_len // NUM_PIC) + 1
        gap = 7 if gap > 6 else gap
        print("For {}, fps => {} and gap => {} ".format(file, fps, gap))
        startPos = 1
        video_len = 0
        bias = 50
        while True:
            success, frame = cap.read()
            if not success:
                break
            video_len += 1
            if gap > 1 and video_len % gap == startPos:
                yy, xx, _ = frame.shape
                left, right = (xx - yy) // 2 - bias, (xx + yy) // 2 + bias
                region = frame[0:yy, left:right]
                if yy < resize_hw:
                    print("Skip the file :" + name)
                    continue
                resROI = cv2.resize(region, (int(resize_hw*1.2), resize_hw), cv2.INTER_AREA)
                frame_path = os.path.join(dir_path, name + "_%d.jpg" % video_len)
                cv2.imwrite(frame_path, resROI)  # save frame as JPEG file
                
        endr = time.time()
        print("Save {} frame to jpeg and elapsed_time is {:.2f}s.".format(video_len // gap, endr - start))
    print("Handle all videos completely.")
    ending = time.time()
    print("Total elapsed_time is {:.2f}s".format(ending-beginning))


if __name__ == '__main__':
    main()
