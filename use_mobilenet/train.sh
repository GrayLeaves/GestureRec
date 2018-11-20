#!/bin/bash
read -t 10 -p "Do you want to export slim into python-path? >> " ans
if [ -n ${ans} -a ${ans} == 'y' ]; then
  export PYTHONPATH="${PYTHONPATH}:/home/zgwu/mobileNetDemo/models:/home/zgwu/mobileNetDemo/models/research/slim/"
  echo "Added slim to python-path ${PYTHONPATH}"
else
  echo "Ok, skip to export the path."
fi

train='train.py'
if [ -e ${train} ]; then
   python ${train}
else
   echo "${train} doesn't exist."
fi

#test='use_model.py'
#if [ -e ${test} ]; then
#   python ${test}
#else
#   echo "${test} doesn't exist."
#fi
