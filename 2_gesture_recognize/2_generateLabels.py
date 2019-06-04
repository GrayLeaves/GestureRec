# -*-coding:utf-8-*-
"""
    @Project: googlenet_classification
    @File   : create_labels_files.py
    @Author : panjq
"""

import os
import os.path
from config import R


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:  # 以空格" "作为分隔符
                    str_line = str_line + str(data) + " "
                else:  # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, dirnames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            # 输出路径下所有文件（包含子文件）信息
            # print(os.path.join(parent, filename))
            curr_file = parent.split(os.sep)[-1]
            # print("Detected the file whose parent dir named " + curr_file)
            if curr_file in R.labelsStr:
                labels = R.labelsDict[curr_file]
                files_list.append([os.path.join(curr_file, filename), labels])
    return files_list


if __name__ == '__main__':
    train_data = get_files_list(R.train_dir)
    write_txt(train_data, R.train_labels, mode='w')
    val_data = get_files_list(R.val_dir)
    write_txt(val_data, R.val_labels, mode='w')

