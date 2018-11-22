## Simple hand gesture recognition
### DataSet
横屏拍摄各类手势居中的视频，隔帧提取并裁剪缩放处理，以减少提取手势的计算量：VideoToImg.py
### Download TensorFlow Models
From: https://github.com/tensorflow/models
export tensorflow/model/research/slim,以便使用nets和mobilenet
### Train own model. (modified from tensorflow_models_nets)
Go to: https://github.com/PanJinquan/tensorflow_models_nets

## MobileNet
### Environment
Ubuntu中搭建强化学习平台（使用anaconda管理Python并安装tensorflow、opencv)
https://www.cnblogs.com/qiangzi0221/p/8331715.html
### LabelImg
https://github.com/tzutalin/labelImg

## Hand3d
### OpenCV face detection
multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
### ColorHandPose3D network
Download : https://github.com/lmb-freiburg/hand3d
### Decription
Use a face detection to narrow the identified area and reduce the amount of recognition calculation.
