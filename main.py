# -*- coding: utf-8 -*-
import os
import torch
import torch.optim
import network as models
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from utils.util import *
from utils.util_args import get_args
from utils.util_loader import data_loader
from utils.util_loss import \
    dsrg_layer, dsrg_seed_loss_layer,\
    softmax_layer, seed_loss_layer,\
    crf_layer, constrain_loss_layer
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.nn as nn

def gpu_allocation(input_tensor, gpu):
    if gpu is not None:
        input_tensor = input_tensor.cuda(gpu)  
    return input_tensor

def grid_prepare(images, outputs, gt_maps, gt_maps_new, true_gt_imgs, batch_size):
    displaynum = 8
    width, height = 100, 100
    size = (width, height)
    images = images.clone().detach().data.cpu()
    outputs = outputs.clone().detach().data.cpu()   
    gt_maps = gt_maps.clone().detach().data.cpu()
    gt_maps_new = gt_maps_new.clone().detach().data.cpu()
    true_gt_imgs = true_gt_imgs.clone().detach().data.cpu()
    
    # writer add_images (origin, output, gt)
    images = images + torch.tensor([123., 117., 107.]).reshape(1, 3, 1, 1)
    images = F.interpolate(images, size=size)
    images = images.type(torch.FloatTensor)
    images_row = vutils.make_grid(images[:displaynum, :,:,:], nrow=displaynum, padding=2, normalize=True, scale_each=True)
    
    true_gt_imgs = F.interpolate(true_gt_imgs, size=size)
    true_gt_imgs = true_gt_imgs.type(torch.FloatTensor)
    true_gt_imgs_row = vutils.make_grid(true_gt_imgs[:displaynum, :,:,:], nrow=displaynum, padding=2, normalize=True, scale_each=True)
    
    outputs_row = make_grid_row(outputs, batch_size, size, width, height, displaynum)
    gt_maps_row = make_grid_row(gt_maps, batch_size, size, width, height, displaynum, normal=False)
    gt_maps_new_row = make_grid_row(gt_maps_new, batch_size, size, width, height, displaynum, normal=False)
       
    outputs = F.interpolate(outputs, size=size)
    outputs = torch.argmax(outputs, dim=1)
    outputs_color = torch.zeros(batch_size, 3, width, height)
    for i in range(batch_size):
        temp = outputs[i,:,:].clone().numpy()
        result = torch.from_numpy(label2rgb(temp).transpose(2, 0, 1))
        outputs_color[i,:,:,:] = result
    outputs = outputs_color.type(torch.FloatTensor)     
    outputs_row = vutils.make_grid(outputs[:displaynum, :,:,:], nrow=displaynum, padding=2, normalize=True, scale_each=True)

    gt_maps = F.interpolate(gt_maps, size=size)
    where_seed = torch.sum(gt_maps, 1)
    _, gt_maps = torch.max(gt_maps, dim=1) 
    gt_maps = torch.where(where_seed>0, gt_maps, torch.tensor([-1]))           
    gt_maps_color = torch.zeros(batch_size, 3, width, height)
    
    for i in range(batch_size):
        temp = gt_maps[i,:,:].clone().numpy()
        result = torch.from_numpy(label2rgb(temp).transpose(2, 0, 1))
        gt_maps_color[i,:,:,:] = result
    gt_maps = gt_maps_color.type(torch.FloatTensor)
    gt_maps_row = vutils.make_grid(gt_maps[:displaynum, :,:,:], nrow=displaynum, padding=2, normalize=True, scale_each=True)
    
    gt_maps_new = F.interpolate(gt_maps_new, size=size)
    where_seed = torch.sum(gt_maps_new, 1)
    _, gt_maps_new = torch.max(gt_maps_new, dim=1) 
    gt_maps_new = torch.where(where_seed>0, gt_maps_new, torch.tensor([-1]))           
    gt_maps_new_color = torch.zeros(batch_size, 3, width, height)
    
    for i in range(batch_size):
        temp = gt_maps_new[i,:,:].clone().numpy()
        result = torch.from_numpy(label2rgb(temp).transpose(2, 0, 1))
        gt_maps_new_color[i,:,:,:] = result
    gt_maps_new = gt_maps_new_color.type(torch.FloatTensor)
    gt_maps_new_row = vutils.make_grid(gt_maps_new[:displaynum, :,:,:], nrow=displaynum, padding=2, normalize=True, scale_each=True)
    
    return images_row, outputs_row, gt_maps_row, gt_maps_new_row, true_gt_imgs_row

def validation(val_images, val_targets, val_gt_maps, val_true_gt_imgs, model, args, num_classes):
    val_images = gpu_allocation(val_images, args.gpu)
    val_targets = gpu_allocation(val_targets, args.gpu)
    val_gt_maps = gpu_allocation(val_gt_maps, args.gpu)
    val_true_gt_imgs = gpu_allocation(val_true_gt_imgs, args.gpu)
    val_outputs, _ = model(val_images)
    val_fc8_softmax = softmax_layer(val_outputs) #prob. bx21x41x41    
    val_gt_map_new = dsrg_layer(val_targets, val_gt_maps, val_fc8_softmax, num_classes, args.thre_fg, args.thre_bg, args.workers)
    val_gt_map_new = gpu_allocation(val_gt_map_new, args.gpu)    
    return val_images, val_true_gt_imgs, val_outputs, val_gt_maps, val_gt_map_new

def print_grad(x):
    print(x.requires_grad, x.is_leaf)

def main():
    args = get_args()
    log_folder = os.path.join('train_log', args.name)
    writer = SummaryWriter(log_folder)
    write_para_report(args)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # number of classes for each dataset.
    if args.dataset == 'PascalVOC':
        num_classes = 21
    elif args.dataset == 'COCO':
        num_classes = 81
    else:
        raise Exception("No dataset named {}.".format(args.dataset))

    # Select Model & Method
    print(args.model)
    model = models.__dict__[args.arch](num_classes=num_classes)
    model = load_model(model, args.resume)   
                
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    optimizer = torch.optim.SGD([
        {'params': get_parameters(model, bias=False, final=False), 'lr':args.lr, 'weight_decay': args.wd},
        {'params': get_parameters(model, bias=True, final=False), 'lr':args.lr * 2, 'weight_decay': 0},
        {'params': get_parameters(model, bias=False, final=True), 'lr':args.lr * 10, 'weight_decay': args.wd},
        {'params': get_parameters(model, bias=True, final=True), 'lr':args.lr * 20, 'weight_decay': 0}
    ], momentum=args.momentum)

    train_loader = data_loader(args)
    data_iter = iter(train_loader)
    train_t = (range(args.max_iter))
    model.train()
    
    val_loader = data_loader(args, debugflag=True)
    val_data_iter = iter(val_loader)
    
    for global_iter in train_t:
        try:
            images, targets, gt_maps, true_gt_imgs = next(data_iter)
        except:
            data_iter = iter(data_loader(args))
            images, targets, gt_maps, true_gt_imgs = next(data_iter)

        images = gpu_allocation(images, args.gpu)
        targets = gpu_allocation(targets, args.gpu)
        gt_maps = gpu_allocation(gt_maps, args.gpu)
        true_gt_imgs = gpu_allocation(true_gt_imgs, args.gpu)

        outputs = model(images)
 
        # boundary loss
        fc8_softmax = softmax_layer(outputs) #prob. bx21x41x41
        fc8_CRF_log = crf_layer(outputs, images, iternum=10)
        loss_c = constrain_loss_layer(fc8_softmax, fc8_CRF_log)  
        
        # seeding loss
        gt_map_new = dsrg_layer(targets, gt_maps, fc8_softmax, num_classes, args.thre_fg, args.thre_bg, args.workers)
        gt_map_new = gpu_allocation(gt_map_new, args.gpu)    
        loss_dsrg, loss_b, loss_bc, loss_f, loss_fc = dsrg_seed_loss_layer(fc8_softmax, gt_map_new)
        
        # total loss
        loss = loss_dsrg + loss_c 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer add_scalars
        writer.add_scalar('loss', loss, global_iter)
        writer.add_scalars('losses', {'loss_dsrg': loss_dsrg, 'loss_c': loss_c}, global_iter)

        with torch.no_grad():
            if global_iter % 20 == 0:
                val_data_iter = iter(val_loader)
                val_images, val_targets, val_gt_maps, val_true_gt_imgs = next(val_data_iter)    

                val_images, val_true_gt_imgs, val_outputs, val_gt_maps, val_gt_map_new\
                    = validation(val_images, val_targets, val_gt_maps, val_true_gt_imgs, model, args, num_classes)
                    
                images_row, outputs_row, gt_maps_row, gt_maps_new_row, true_gt_imgs_row\
                    = grid_prepare(val_images, val_outputs, val_gt_maps, val_gt_map_new, val_true_gt_imgs, 8)
 
                images_row, outputs_row, gt_maps_row, gt_maps_new_row, true_gt_imgs_row\
                    = grid_prepare(images, outputs, gt_maps, gt_map_new, true_gt_imgs, args.batch_size)
         
                grid_image = torch.cat((images_row, true_gt_imgs_row, outputs_row, gt_maps_row, gt_maps_new_row), dim=1)
                writer.add_image(args.name, grid_image, global_iter)
                writer.close()
        
        description = "[{0:4d}/{1:4d}] loss: {2:.3f} dsrg: {3:.3f} bg: {5:.3f} fg: {6:.3f} c: {4:.3f}".format(global_iter+1, \
                        args.max_iter, loss, loss_dsrg, loss_c, loss_b, loss_f)

        print(description)        

        # save snapshot
        if global_iter % args.snapshot == 0:
            save_checkpoint(model.state_dict(), log_folder, 'checkpoint_%d.pth.tar' % global_iter)

        # lr decay
        if global_iter % args.lr_decay == 0:
            args.lr = args.lr * 0.1
            optimizer = adjust_learning_rate(optimizer, args.lr)

    print("Training is over...")
    save_checkpoint(model.state_dict(), log_folder, 'last_checkpoint.pth.tar')

if __name__ == '__main__':
    main()