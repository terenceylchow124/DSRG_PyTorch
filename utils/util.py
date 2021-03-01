import skimage.color as imgco
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import argparse
import copyreg
import types
import cv2
import os

palette = np.array([(0, 0, 0), (128, 0, 0), (0, 128,0 ), (128, 128, 0),
                  (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                  (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                  (0, 64, 128)]) # using palette for pascal voc

def make_grid_row(images, batch_size, size, width, height, displaynum, normal=True):
    images = F.interpolate(images, size=size)
    
    if normal==True:
        images = torch.argmax(images, dim=1)
    else:
        where_seed = torch.sum(images, 1)
        _, images = torch.max(images, dim=1) 
        images = torch.where(where_seed>0, images, torch.tensor([-1]))           
   
    images_color = torch.zeros(batch_size, 3, width, height)
    
    for i in range(batch_size):
        temp = images[i,:,:].clone().numpy()
        result = torch.from_numpy(label2rgb(temp).transpose(2, 0, 1))
        images_color[i,:,:,:] = result
    images = images_color.type(torch.FloatTensor)     
    images_row = vutils.make_grid(images[:displaynum, :,:,:], \
                                  nrow=displaynum, padding=2, \
                                  normalize=True, scale_each=True)
    return images_row

def label2rgb(label,colors=[],ignore_label=128,ignore_color=(255,255,255)):
    height, width = label.shape
    if len(colors) <= 0:
        index = np.unique(label)
        index = index[index<21]
        label_mask = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                if label[i,j]==-1:
                    label_mask[i,j]=ignore_color
                else:
                    label_mask[i,j]=palette[label[i,j]]
                    
    return label_mask.astype(np.float)

def get_parameters(model, bias=False,final=False):
    if final:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 21:
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if not m.out_channels == 21:
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight

def save_checkpoint(state, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))

def load_model(model, model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    origin_state_dict = model.state_dict()
    model.load_state_dict(checkpoint, strict=False)
    return model
    
def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr']=2*lr
    optimizer.param_groups[2]['lr']=10*lr
    optimizer.param_groups[3]['lr']=20*lr
    return optimizer

def write_para_report(args):
    
    result_path = os.path.join('train_log', args.name)
    report_name = result_path + "/para_list.txt"
    with open(report_name, 'w') as f:
        f.write("{0:12s}: {1:20s}\n".format('result_path', result_path))
        f.write('{0:12s}: {1:.4f}\n'.format('BS', args.batch_size ))
        f.write("{0:12s}: {1:.4f}\n".format('Iteration', args.max_iter))
        f.write("{0:12s}: {1:.4f}\n".format('lr', args.lr))
        f.write("{0:12s}: {1:.4f}\n".format('wd', args.wd))
        f.write("{0:12s}: {1:.4f}\n".format('step size', args.lr_decay))

        f.write("\nDSRG\n")
        f.write("{0:12s}: {1:.4f}\n".format('thre_fg', args.thre_fg))
        f.write("{0:12s}: {1:.4f}\n".format('thre_bg', args.thre_bg))
        

