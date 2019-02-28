#!/bin/bash
# source activate tensorflow >>> tf

export PYTHONPATH="${PYTHONPATH}:/home/zgwu/mobileNetDemo/models:/home/zgwu/mobileNetDemo/handtracking/images/slim/"
echo " export slim to ${PYTHONPATH} ok."

graph=/home/zgwu/mobileNetDemo/models/research/object_detection/export_inference_graph.py
if [ -e ${graph} ]; then
   python ${graph} --input_type image_tensor --pipeline_config_path ssd_mobilenet_v1_coco.config  \
       --trained_checkpoint_prefix ./tfrecords/model.ckpt-14779 --output_directory ./inference_graph/
else
   echo "${graph} doesn't exist."
fi