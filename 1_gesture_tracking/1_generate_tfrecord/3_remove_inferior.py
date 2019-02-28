# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 18:00:11 2019

@author: zgwu
"""
import os
import pandas as pd

target_path = './jpegs/'
TYPE = 'train'
diff_path = os.path.join(target_path, '{}_remove/'.format(TYPE))
src_csv = os.path.join(target_path, '{}.csv'.format(TYPE))
dst_csv = os.path.join(target_path, '{}_m.csv'.format(TYPE))


def main():
    assert os.path.isdir(diff_path), '{} must be dir'.format(diff_path)
    diff_set = set()
    for file in os.listdir(diff_path):
        if file.endswith('.jpg'):
            filename = file.split(os.sep)[-1]
            name, ext = os.path.splitext(filename)
            diff_set.add(name[:-2] + ext)
    df = pd.read_csv(src_csv, iterator=True) 
    loop = True
    chunk_size = 100
    select_chunk = []
    while loop:
        try:
            chunk = df.get_chunk(chunk_size)
            select_chunk.append(chunk[chunk['filename'].map(lambda item: not(item in diff_set))])
        except StopIteration:
            loop = False
    select_row = pd.concat(select_chunk, ignore_index = True)
    select_row.to_csv(dst_csv, index=False)
    
    
if __name__ == '__main__':
    main()