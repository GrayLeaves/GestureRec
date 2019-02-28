### Simple hand-gesture-recognition using mobilenet

#### Environment
* 详见[Ubuntu中搭建强化学习平台（使用anaconda管理Python并安装tensorflow、opencv)
](https://www.cnblogs.com/qiangzi0221/p/8331715.html)
* [A]. DownLoad [TensorFlow Models](https://github.com/tensorflow/models) 
* [B]. Using [LabelImg](https://github.com/tzutalin/labelImg) 作为标定工具
* [C]. Using [ColorHandPose3D network](https://github.com/lmb-freiburg/hand3d) 作为生成数据的辅助工具
* [D]. Reference [Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow](https://github.com/victordibia/handtracking) & [tensorflow_models_nets](https://github.com/PanJinquan/tensorflow_models_nets)

#### 0. Experience
    From https://github.com/Itseez/opencv/tree/master/data/haarcascades download multiple cascades.
    Use a face detection to narrow the identified area and reduce the amount of recognition calculation.
    其中目录0中的run_detect_single_hand_3d.py和run_detect_single_hand_ssd.py用于测试运行下载的项目C和D.
#### 1. gesture tracking
    抽取视频各帧图片，从中精准抠出手势，制作tfrecord文件，训练并生成实时跟踪手势的模型。
#### 2. gesture recognize
    将抠取的手势图分类并训练出分类器，以提高手势识别精准度。