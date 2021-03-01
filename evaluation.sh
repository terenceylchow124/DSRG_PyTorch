#!/bin/bash
image_list='./datalist/PascalVOC/val_id.txt'
gt_path="./dataset/PascalVOC/VOCdevkit/VOC2012/SegmentationClassAug/"
result_path='./result/1'
pred_path=${result_path}/label_mask
save_name=${result_path}/evaluation_result_direct.txt
color_mask=0

python evaluation.py \
  --image-list ${image_list} \
  --pred-path ${pred_path} \
  --gt-path ${gt_path} \
  --save-name ${save_name} \
  --class-num 21 \
  --color-mask ${color_mask} 