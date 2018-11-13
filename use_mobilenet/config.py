# coding=utf-8
import datetime


class R(object):
    """
        General settings
    """
    # generateLabels
    labelsStr = ['Fist', 'Admire', 'Victory', 'Okay', 'None', 'Palm', 'Hex']
    labelsDict = {'Fist': 0, 'Admire': 1, 'Victory': 2, 'Okay': 3, 'None': 4, 'Palm': 5, 'Hex': 6}
    train_dir = './support_data/images/train/'
    val_dir = './support_data/images/validation/'
    test_dir = './support_data/images/test/'

    # pic2tfrecords // generateTfrecords
    # 产生train.record文件
    train_labels = './support_data/records_txt/train.txt'
    train_record_output = './support_data/tfrecords/train_1113.tfrecords'
    # 产生val.record文件
    val_labels = './support_data/records_txt/val.txt'
    val_record_output = './support_data/tfrecords/val_1113.tfrecords'

    # train.py
    dat = datetime.date.today()
    model_dir = './weights/{:2d}{:2d}/'.format(dat.month, dat.day)
    num_classes = len(labelsStr)
    batch_size = 16
    resize_height = 300  # 指定存储图片高度
    resize_width = 300   # 指定存储图片宽度
    depths = 3
    log_path = 'recordLog/'

    data_shape = [batch_size, resize_height, resize_width, depths]
    train_record_file = train_record_output
    val_record_file = val_record_output

    base_lr = 0.01  # 学习率
    max_steps = 20000  # 迭代次数
    
    train_log_step = max_steps // 100
    val_log_step = max_steps // 25
    
    snapshot = max_steps // 5  # 保存文件间隔
    snapshot_prefix = model_dir + 'model.ckpt'

    # use_model
    image_dir = './support_data/images/test/'
    labels_filename = './support_data/label.txt'
    models_path = snapshot_prefix + '-' + str(max_steps)
