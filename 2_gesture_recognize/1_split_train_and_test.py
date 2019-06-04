# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 19:39:27 2018

@author: wuzhenguang
"""

import os
import shutil
import random
from config import R


def move_img(train_path, test_path, *, factor=0.8):
    if not os.path.isdir(train_path):
        raise NotADirectoryError(train_path + " doesn't exist.")
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    num = 0
    for root, dirnames, _ in os.walk(train_path):  # root path
        for subDirName in dirnames:  # enter sub_path
            src_list, dst_list = [], []
            fullDirPath = os.path.join(root, subDirName)
            fullTargetPath = os.path.join(test_path, subDirName)
            if os.path.isdir(fullTargetPath):
                os.rmdir(fullTargetPath)    # ensure it as empty dir
            os.mkdir(fullTargetPath)
            for _, _, filenames in os.walk(fullDirPath):  # full_sub_path
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        src_list.append(filename)  # all files
                num_file = int((1-factor) * len(filenames))
                print("Plan move {} files from {} ...".format(num_file, subDirName))
                dst_list = random.sample(src_list, num_file)  # just record filename
            for filename in dst_list:
                src = os.path.join(fullDirPath, filename)
                dst = os.path.join(fullTargetPath, filename)
                shutil.move(src, dst)
                num += 1
    print("Ok, move " + str(num) + " of files from train as validation.")


def restore_img(train_path, test_path):
    if not os.path.isdir(test_path):
        raise NotADirectoryError(train_path + " doesn't exist.")
    if not os.path.isdir(test_path):
        raise NotADirectoryError(test_path + " doesn't exist.")

    num = 0
    for root, dirnames, _ in os.walk(test_path):  # root path
        for subDirName in dirnames:  # enter sub_path
            src_list = []
            fullDirPath = os.path.join(root, subDirName)
            fullTargetPath = os.path.join(train_path, subDirName)
            for _, _, filenames in os.walk(fullDirPath):  # full_sub_path
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        src_list.append(filename)  # all files
            for filename in src_list:
                src = os.path.join(fullDirPath, filename)
                dst = os.path.join(fullTargetPath, filename)
                # print('From' + src + 'to' + dst)
                shutil.move(src, dst)
                num += 1
    print("Ok, Restore {} test files together.".format(num))


if __name__ == '__main__':
    restore_img(R.train_dir, R.val_dir)
    move_img(R.train_dir, R.val_dir, factor=0.85)
