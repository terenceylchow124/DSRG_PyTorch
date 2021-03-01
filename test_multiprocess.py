import os
import cv2
import pylab
import warnings
import argparse
import numpy as np
import scipy.ndimage as nd
from multiprocessing import Process
import torch.nn.functional as F
import torch
import network as models
from utils.util import load_model
import krahenbuhl2013
import skimage.color as imgco
import skimage.io as imgio
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

IMAGE_MEAN_VALUE = [104.0, 117.0, 123.0]

label2rgb_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128,0 ), (128, 128, 0),
                  (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                  (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                  (0, 64, 128)]) # using palette for pascal voc

def label2rgb(label,cls_label,colors=[],ignore_label=128,ignore_color=(255,255,255)):

    if len(colors) <= 0:
        index = np.unique(label)
        index = index[index<21]
        colors = label2rgb_colors[index]
    label = imgco.label2rgb(label,colors=colors,bg_label=ignore_label,bg_color=ignore_color)
    return label.astype(np.uint8)

def parser_args():
    parser = argparse.ArgumentParser(description='Get segmentation prediction')
    parser.add_argument("--image-list", default='./datalist/PascalVOC/val_id.txt', type=str, help="Path to image list")
    parser.add_argument("--image-path", default='./dataset/PascalVOC/VOCdevkit/VOC2012',type=str, help="Path to image")
    parser.add_argument('--cls-labels-path', default='./datalist/PascalVOC/cls_labels.npy', type=str)
    parser.add_argument("--arch", default='deeplab_large_fov', type=str, help="Model type")
    parser.add_argument("--trained", default='./train_log/1', type=str, help="Model weights")
    parser.add_argument("--pred-path", default='./result/1', type=str, help="Output png file name")
    parser.add_argument("--smooth", action='store_true', help="Apply postprocessing")
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int, help='GPU device id to use [0]')
    parser.add_argument('--split-size', default=8, type=int)
    parser.add_argument('--num-gpu', default=1, type=int)
    parser.add_argument('--color-mask', type=int, default=1)

    args = parser.parse_args()
    return args

def preprocess(image, size, mean_pixel):
    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]), size / float(image.shape[1]), 1.0),
                    order=1)

    # RGB to BGR
    image = image[:, :, [2, 1, 0]]
    image = image - np.array(mean_pixel)

    # BGR to RGB
    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)

def predict_label_mask(image_file, model, smooth, gpu_id):
    im = pylab.imread(image_file)

    image = torch.from_numpy(preprocess(im, 321, IMAGE_MEAN_VALUE).astype(np.float32))
    image = image.cuda(gpu_id)
    featmap, _ = model(image)
    scores = featmap.reshape(21, 41, 41).detach().cpu().numpy().transpose(1, 2, 0)
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
    probs = nd.zoom(probs, (d1 / probs.shape[0], d2 / probs.shape[1], 1.0), order=1)
    probs1 = probs

    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
    else:
        result = np.argmax(probs, axis=2)

    return result 

def predict_color_mask(image_file, model, smooth, gpu_id, cls_label):
    cls_label = np.insert(cls_label, 0, 1)
    cls_label = np.squeeze(np.asarray(np.nonzero(cls_label), dtype=int))
    im = pylab.imread(image_file)
    d1, d2 = float(im.shape[0]), float(im.shape[1])
    image = torch.from_numpy(preprocess(im, 321, IMAGE_MEAN_VALUE).astype(np.float32))
    image = image.cuda(gpu_id)
    
    featmap = model(image)
                      
    featmap = featmap.reshape(21, 41, 41).detach().cpu().numpy().transpose(1, 2, 0)
    featmap = nd.zoom(featmap, (d1 / featmap.shape[0], d2 / featmap.shape[1], 1.0), order=2)

    if smooth:
        crf_pred = krahenbuhl2013.CRF(im, np.array(featmap), scale_factor=1.0)
    else:
        crf_pred = featmap
    
    output = label2rgb(np.argmax(featmap,axis=2), cls_label)
    pred = label2rgb(np.argmax(crf_pred,axis=2), cls_label)

    return output, pred


def save_mask_multiprocess(num, data_size):
    process_id = os.getpid()
    print('process {} starts...'.format(process_id))

    if args.num_gpu == 1:
        gpu_id = args.gpu_id
    elif args.num_gpu == 2:
        if num >= data_size // args.num_gpu:
            gpu_id = args.gpu_id + 0
        else:
            gpu_id = args.gpu_id + 1
    elif args.num_gpu == 4:
        if num >= data_size // args.num_gpu * 3:
            gpu_id = args.gpu_id + 0
        elif num >= data_size // args.num_gpu * 2:
            gpu_id = args.gpu_id + 1
        elif num >= data_size // args.num_gpu * 1:
            gpu_id = args.gpu_id + 2
        else:
            gpu_id = args.gpu_id + 3
    else:
        raise Exception("ERROR")
    
    base_model = models.__dict__[args.arch](num_classes=21)
    model = base_model    
    model = load_model(model, args.trained)    
    model = model.cuda(gpu_id)
    model.eval()      
        
    if num == data_size - 1:
        sub_image_ids = image_ids[num * len(image_ids) // data_size:]
    else:
        sub_image_ids = image_ids[num * len(image_ids) // data_size: (num + 1) * len(image_ids) // data_size]
    if num == 0:
        print(len(sub_image_ids), 'images per each process...')

    for idx, img_id in enumerate(sub_image_ids):
        if num == 0 and idx % 10 == 0:
            print("[{0} * {3}]/[{1} * {3}] : {2} is done.".format(idx, len(sub_image_ids), img_id, args.split_size))
        image_file = os.path.join(image_path, img_id + '.jpg')
        cls_label = cls_list[img_id]

        if args.color_mask:
            output, pred = predict_color_mask(image_file, model, args.smooth, gpu_id, cls_label)
            save_path = os.path.join(args.pred_path, "output" ,img_id + '_output.png')
            cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            save_path = os.path.join(args.pred_path, "pred", img_id + '_pred.png')
            cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        else:
            labelmap = predict_label_mask(image_file, model, args.smooth, gpu_id)
            save_path = os.path.join(args.pred_path, "label_mask" ,img_id + '_labelmask.png')
            cv2.imwrite(save_path, labelmap)

if __name__ == "__main__":
    args = parser_args()
    image_ids = [i.strip() for i in open(args.image_list) if not i.strip() == '']
    image_path = os.path.join(args.image_path, 'JPEGImages')

    if args.pred_path and (not os.path.isdir(args.pred_path)):
        os.makedirs(args.pred_path)
        
    pred = args.pred_path + "/pred"
    output = args.pred_path + "/output"
    label_mask = args.pred_path + "/label_mask"

    if not os.path.isdir(pred):
        os.makedirs(pred)
    if not os.path.isdir(output):
        os.makedirs(output)
    if not os.path.isdir(label_mask):
        os.makedirs(label_mask)

    cls_list = np.load(args.cls_labels_path, allow_pickle=True).tolist()
    split_size = args.split_size * args.num_gpu
    numbers = range(split_size)
    processes = []
    for index, number in enumerate(numbers):
        proc = Process(target=save_mask_multiprocess, args=(number, split_size,))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()
