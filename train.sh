#!/bin/bash
gpu=1
name=1
arch=deeplab_large_fov
resume='./datalist/PascalVOC/vgg16_20M_custom.pth'
dataset="PascalVOC"
data='./dataset/PascalVOC/VOCdevkit/VOC2012'
gt_root='./datalist/PascalVOC/localization_cues.pickle'
train_list='./datalist/PascalVOC/input_list.txt'
batch=20
gamma=0.1
max_iter=8000
stepsize=2000
snapshot=500
wd=0.0005
lr=0.0005
thre_fg=0.85
thre_bg=0.99

python3 main.py \
    --arch ${arch} \
    --name ${name} \
    --data ${data} \
    --gt-root ${gt_root} \
    --dataset ${dataset} \
    --train-list ${train_list} \
    --max-iter ${max_iter} \
    --snapshot ${snapshot} \
    --lr-decay ${stepsize} \
    --batch-size ${batch} \
    --lr ${lr} \
    --wd ${wd} \
    --resume ${resume} \
    --thre-bg ${thre_bg} \
    --thre-fg ${thre_fg}\
    --gpu ${gpu}
