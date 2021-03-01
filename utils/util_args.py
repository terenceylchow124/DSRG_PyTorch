# -*- coding: utf-8 -*-
import argparse
import network as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch PascalVOC Training')
    parser.add_argument('--name', type=str, default='1')
    parser.add_argument('--evaluate', action='store_true', help='evaluate mode')
    parser.add_argument('--arch', default='deeplab_large_fov', choices=model_names, help='model choice')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument("--resume", default="./datalist/PascalVOC/vgg16_20M_custom.pth", type=str, help='checkpoint path to resume')
    parser.add_argument('--snapshot', default=500, type=int, help='snapshot point')

    # hyperparamter
    parser.add_argument('--max-iter', type=int, default=8000, help='number of total iteration to run')
    parser.add_argument('--lr-decay', type=int, default=2000, help='Reducing lr frequency')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--batch-size', default=20, type=int,  help='mini-batch size')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--nest', action='store_true', help='nestrov for optimizer')

    # path
    parser.add_argument('--data', default='./dataset/PascalVOC/VOCdevkit/VOC2012', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='PascalVOC', )
    parser.add_argument('--gt-root', type=str, default='./datalist/PascalVOC/localization_cues.pickle')
    parser.add_argument('--train-list', type=str, default='./datalist/PascalVOC/input_list.txt')

    # data transform
    parser.add_argument('--resize-size', type=int, default=321, help='input resize size')
    parser.add_argument('--crop-size', type=int, default=321, help='input crop size')

    parser.add_argument('--debug', default=False, action='store_true')
    
    # dsrg
    parser.add_argument('--model', type=int, default=1, help='deepLab model version')
    parser.add_argument('--thre-fg', type=float, default=0.5, help='foreground threshold')
    parser.add_argument('--thre-bg', type=float, default=0.7, help='background threshold')
         
    args = parser.parse_args()
  
    return args
