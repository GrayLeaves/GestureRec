#!/bin/bash
# source activate tensorflow >>> tf

export PYTHONPATH="${PYTHONPATH}:/home/zgwu/models:/home/zgwu/slim/"
echo " export slim to ${PYTHONPATH} ok."

eval=/home/zgwu/models/research/object_detection/legacy/eval.py  # legacy/train.py
if [ -e ${eval} ]; then
   python ${eval} --logtostderr --eval_dir ./test_log/ --checkpoint_dir=./tfrecords/ --pipeline_config_path ssd_mobilenet_v1_coco.config
else
   echo "${eval} doesn't exist."
fi