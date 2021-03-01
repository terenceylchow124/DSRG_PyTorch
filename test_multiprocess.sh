#!/bin/bash
arch=deeplab_large_fov
image_list='./datalist/PascalVOC/val_id.txt'
image_path='./dataset/PascalVOC/VOCdevkit/VOC2012'
cls_labels_path='./datalist/PascalVOC/cls_labels.npy'
log_path='./train_log/1'
pred_path='./result/1'
trained=${log_path}/last_checkpoint.pth.tar
smooth=True
color_mask=1
gpu=0

python3 test_multiprocess.py \
  --arch ${arch} \
  --trained ${trained} \
  --image-list ${image_list} \
  --image-path ${image_path} \
  --pred-path ${pred_path} \
  --cls-labels-path ${cls_labels_path} \
  --gpu ${gpu} \
  --num-gpu 1 \
  --split-size 8 \
  --smooth \
  --color-mask ${color_mask} \
