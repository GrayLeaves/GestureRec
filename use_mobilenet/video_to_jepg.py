# coding=utf-8
import cv2
import os
import time
import shutil
from PIL import Image

resize_hw = 300
video_path = '../video/1106/'
save_path = '../video_img/1107_jpg/'


def main():
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
        raise FileNotFoundError(video_path + " doesn't exist.")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    num = 0
    for video_name in video_list:
        num += 1
        file = video_name.split("/")[-1]  # extract the filename
        name, ext = os.path.splitext(file)  # get rid of ext
        dir_path = save_path + name + "/"  # frame save_path
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        resize_path = save_path + name + "_resize/"
        if not os.path.isdir(resize_path):
            os.mkdir(resize_path)

        print("Process the video : " + name + " ... {}/{}".format(num, len(video_list)))
        start = time.time()
        cap = cv2.VideoCapture(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("The fps of " + file + " is {:.2f}/s".format(fps))
        gap = fps // 6  # each frame between 0.167s
        startPos = 3
        video_len = 0
        success = True
        image_list = list()
        while success:
            success, image = cap.read()
            video_len += 1
            if video_len % gap == startPos:
                frame_path = dir_path + name + "_%d.jpg" % (video_len // gap)
                cv2.imwrite(frame_path, image)  # save frame as JPEG file
                image_list.append(frame_path)

        endr = time.time()
        print("Save all frame to jpeg and elapsed_time is {:.2f}s.".format(endr - start))

        cnt = 0
        # Feed image list through network
        for img_name in image_list:
            cnt += 1
            im = Image.open(img_name)
            filename = img_name.split("/")[-1]
            xx, yy = im.size
            left, right = (xx - yy) // 2, (xx + yy) // 2
            up, down = 0, yy
            size = (left, up, right, down)
            region = im.crop(size)
            if yy < resize_hw:
                print("Skip the file :" + filename)
                continue
            square = region.resize((resize_hw, resize_hw), Image.ANTIALIAS)
            square.save(resize_path + filename)
            if cnt == video_len // gap:
                break

        ends = time.time()
        print("Saved all resize picture and elapsed_time is {:.2f}s.".format(ends - endr))
        shutil.rmtree(dir_path)
        '''
        fps = 24
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter(video_path + name + '_res.avi', fourcc, fps, (resize_hw, resize_hw))
        print("Let's view the video.")
        for i in range(video_len // gap + 1):
            frame = cv2.imread(resize_path + str(i) + '.png')
            videoWriter.write(frame)
        videoWriter.release()
        endv = time.time()
        print("Save it as video and elapsed_time is {:.2f}s".format(endv-ends))
        '''
    print("Handle all videos completely.")
    ending = time.time()
    print("Total elapsed_time is {:.2f}s".format(ending-beginning))

if __name__ == '__main__':
    main()
