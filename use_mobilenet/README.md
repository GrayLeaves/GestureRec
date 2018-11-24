## Simple hand-gesture-recognition using mobilenet

### Environment
Ubuntu中搭建强化学习平台（使用anaconda管理Python并安装tensorflow、opencv)
https://www.cnblogs.com/qiangzi0221/p/8331715.html
### LabelImg
https://github.com/tzutalin/labelImg

### Download TensorFlow Models
https://github.com/tensorflow/models
export tensorflow/model/research/slim,以便使用nets和mobilenet

### DataSet
#### (1) Using [ColorHandPose3D network]
https://github.com/lmb-freiburg/hand3d
#### OpenCV face detection
multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
Use a face detection to narrow the identified area and reduce the amount of recognition calculation.
横屏拍摄各类手势居中的视频，隔帧提取并裁剪缩放处理，以减少提取手势的计算量：<br>
推荐GPU上运行: ```python hand3d/run_detect_single_hand_3d.py ```
#### (2) Using [Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow]
https://github.com/victordibia/handtracking
横屏拍摄各类手势视频，隔帧提取并缩放处理：<br>
可在CPU上运行: ```python handtracking/run_detect_single_hand_ssd.py```

### Train your own model
Modified from tensorflow_models_nets: https://github.com/PanJinquan/tensorflow_models_nets
