#!/bin/bash
read -t 10 -p "Do you want to export slim into python-path(y/n)? >> " ans
if [ -n ${ans} -a ${ans} == 'y' ]; then
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/models:$(pwd)/models/research/slim/"
  echo "Export slim into ${PYTHONPATH}"
else
  echo "Ok, skip anyways."
fi

train='train.py'
if [ -e ${train} ]; then
   python ${train}
else
   echo "${train} doesn't exist."
fi