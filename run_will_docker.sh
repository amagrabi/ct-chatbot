#!/bin/bash

if [ -z ${MODEL_SOURCE_TYPE} ]
then
    MODEL_SOURCE_TYPE=bucket
    MODEL_SOURCE_BUCKET=ctp-playground-ml-datasets
    MODEL_SOURCE_BUCKET_DIR=hipchat/models
fi

if [ "${MODEL_SOURCE_TYPE}" == "bucket" ]
then
    rm -rf /mnt
    mkdir /mnt
    mkdir /mnt/models
    /usr/bin/gcsfuse --implicit-dirs --only-dir ${MODEL_SOURCE_BUCKET_DIR} ${MODEL_SOURCE_BUCKET} /mnt/models

    rm -rf models
    mkdir models
    cp -r /mnt/models/* models

    fusermount -u /mnt/models
    rm -rf /mnt
fi

if [ "${MODEL_SOURCE_TYPE}" == "folder" ]
then
    rm -rf models
    mkdir models
    cp -r ${MODEL_SOURCE_FOLDER}/* models
fi

./run_will.py