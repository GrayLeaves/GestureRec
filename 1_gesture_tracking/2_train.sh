#!/bin/bash
# source activate tensorflow >>> tf

export PYTHONPATH="${PYTHONPATH}:/home/zgwu/models:/home/zgwu/slim/"
echo " export slim to ${PYTHONPATH} ok."

train=/home/zgwu/models/research/object_detection/legacy/train.py  # or model_main.py
if [ -e ${train} ]; then
   python ${train}  --logtostderr --train_dir ./hand_record/01_10/ --pipeline_config_path ssd_mobilenet_v1_coco.config
else
   echo "${train} doesn't exist."
fi
echo "Train Completed."
