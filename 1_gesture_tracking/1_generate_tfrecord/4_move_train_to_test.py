#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:52:15 2019

@author: zgwu
"""
import os
import random


def mv_img(src_path, dst_path, factor):
    files = os.listdir(src_path)
    loop_index = 0
    data_size = len(files)
    data_sampsize = int(factor * data_size)
    test_samp_array = random.sample(range(data_size), data_sampsize)
    for f in files:
        if(f.endswith('.jpg')):
            loop_index += 1
            if loop_index in test_samp_array:
                os.rename(os.path.join(src_path,f), os.path.join(dst_path,f))
    print('Move {} files from {} to {}.'.format(data_sampsize, src_path, dst_path))


if __name__ == '__main__':
    train = './train/'
    test = './test/'
    mv_img(train, test, 0.1)