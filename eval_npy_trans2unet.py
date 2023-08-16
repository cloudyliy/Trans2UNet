# CUDA_VISIBLE_DEVICES=0 python eval_npy_trans2unet.py
import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.optim import lr_scheduler

from utils_casnet import get_images
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
# from logger import Logger
import os

from tqdm import tqdm
import cv2


from cascade_transunet_concate import cascadTUNet

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_curve, auc, f1_score
from torchsummaryX import summary

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = OptionParser()
parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                  type='int', help='batch size')
# parser.add_option('-p', '--log-dir', dest='logdir', default='eval',
#                     type='str', help='tensorboard log')
# parser.add_option('-m', '--model', dest='model', default='MODEL.pth.tar',
#                     type='str', help='models stored')
parser.add_option('-m', '--model', dest='model',
                  default='/home/DR/results/models/Trans2UNet/model_AP.pth.tar',
                  type='str', help='models stored')
parser.add_option('-n', '--net-name', dest='netname', default='cascadTUNet',
                  type='str', help='net name, unet or cascadTUNet')
# parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
#                       default=False, help='preprocess input images')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                  default='1', help='preprocess input images')
# parser.add_option('-i', '--healthy-included', dest='healthyincluded', action='store_true',
#                       default=False, help='include healthy images')

(args, _) = parser.parse_args()

# logger = Logger('./logs', args.logdir)
net_name = args.netname
lesions = ['ex', 'he', 'ma', 'se']



image_dir = '/home/DR/data'

# logdir = args.logdir
# if not os.path.exists(logdir):
#     os.mkdir(logdir)


if net_name == 'cascadTUNet':
    logdir = 'DR/eval_npy/Trans2unet/'
    figure_out_dir = 'DR/figure/Trans2unet/'
    args.model = 'DR/results/models/Trans2UNet/model_AP.pth.tar'


if not os.path.exists(logdir):
    os.makedirs(logdir)


if not os.path.exists(figure_out_dir):
    os.makedirs(figure_out_dir)

figure_out_dir_EX = os.path.join(figure_out_dir,'EX/')
figure_out_dir_SE = os.path.join(figure_out_dir,'SE/')
figure_out_dir_HE = os.path.join(figure_out_dir,'HE/')
figure_out_dir_MA = os.path.join(figure_out_dir,'MA/')

if not os.path.exists(figure_out_dir_EX):
    os.makedirs(figure_out_dir_EX)
if not os.path.exists(figure_out_dir_SE):
    os.makedirs(figure_out_dir_SE)
if not os.path.exists(figure_out_dir_HE):
    os.makedirs(figure_out_dir_HE)
if not os.path.exists(figure_out_dir_MA):
    os.makedirs(figure_out_dir_MA)

softmax = nn.Softmax(1)


def eval_model(model, eval_loader):
    model.to(device=device)
    model.eval()
    eval_tot = len(eval_loader)
    vis_images = []

    with torch.set_grad_enabled(False):
        batch_id = 0
        for inputs, true_masks in tqdm(eval_loader):

            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape


            if net_name == 'cascadTUNet':
                x1,x2,x3,x4,x5,x6 = model(inputs)
                masks_pred = x6



            masks_pred_sigmoid = torch.sigmoid(masks_pred)
            masks_soft = masks_pred_sigmoid[:, :, :, :].to("cpu")

            true_masks = torch.where(true_masks[:, :, :, :] > 0.5, 1, 0)
            true_masks = true_masks.to("cpu")

            masks_hard = torch.zeros(masks_pred_sigmoid.shape).to(dtype=torch.float, device=inputs.device)
            n_number=4

# ##0.5
#             for i in range(n_number):
#
#                 mask_predict_avg_binary = torch.where(masks_pred_sigmoid[:, i, :, :] > 0.5, 1, 0)
#                 masks_hard[:,i] = mask_predict_avg_binary
#             masks_hard = masks_hard.to("cpu")

##f1
            for i in range(n_number):
                precision, recall, thresholds = precision_recall_curve(true_masks[:, i+1, :, :].reshape(-1), masks_soft[:, i, :, :].reshape(-1))

                f1_scores = 2 * recall * precision / (recall + precision)
                best_f1_th = thresholds[np.argmax(f1_scores)]

                mask_predict_avg_binary = torch.where(masks_pred_sigmoid[:, i, :, :] > best_f1_th, 1, 0)

                masks_hard[:,i] = mask_predict_avg_binary
            masks_hard = masks_hard.to("cpu")



            np.save(os.path.join(logdir, 'mask_soft_' + str(batch_id) + '.npy'), masks_soft.numpy())
            np.save(os.path.join(logdir, 'mask_true_' + str(batch_id) + '.npy'), true_masks[:, 1:].numpy())

            np.save(os.path.join(logdir, 'mask_hard_' + str(batch_id) + '.npy'), masks_hard.numpy())
            true_GT = true_masks[:, 1:].cpu().numpy()

            cv2.imwrite(figure_out_dir_EX + str(batch_id) + '_EX.png', masks_hard[0, 0, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_EX + str(batch_id) + '_soft_EX.png', masks_soft[0, 0, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_EX + str(batch_id) + '_GT_EX.png', true_GT[0, 0, :, :] * 255)

            cv2.imwrite(figure_out_dir_SE + str(batch_id) + '_SE.png', masks_hard[0, 1, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_SE + str(batch_id) + '_soft_SE.png', masks_soft[0, 1, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_SE + str(batch_id) + '_GT_SE.png', true_GT[0, 1, :, :] * 255)

            cv2.imwrite(figure_out_dir_HE + str(batch_id) + '_HE.png', masks_hard[0, 2, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_HE + str(batch_id) + '_soft_HE.png', masks_soft[0, 2, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_HE + str(batch_id) + '_GT_HE.png', true_GT[0, 2, :, :] * 255)

            cv2.imwrite(figure_out_dir_MA + str(batch_id) + '_MA.png', masks_hard[0, 3, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_MA + str(batch_id) + '_soft_MA.png', masks_soft[0, 3, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_MA + str(batch_id) + '_GT_MA.png', true_GT[0, 3, :, :] * 255)


            batch_id += 1
    return  vis_images


def denormalize(inputs):
    if net_name == 'unet':
        return (inputs * 255.).to(device=device, dtype=torch.uint8)
    else:
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        return ((inputs * std[None, :, None, None] + mean[None, :, None, None]) * 255.).to(device=device,
                                                                                           dtype=torch.uint8)


def generate_log_images_full(inputs, true_masks, masks_soft, masks_hard):
    true_masks = (true_masks * 255.).to(device=device, dtype=torch.uint8)
    masks_soft = (masks_soft * 255.).to(device=device, dtype=torch.uint8)
    masks_hard = (masks_hard * 255.).to(device=device, dtype=torch.uint8)
    inputs = denormalize(inputs)
    bs, _, h, w = inputs.shape
    pad_size = 5
    images_batch = (torch.ones((bs, 3, h * 2 + pad_size, w * 2 + pad_size)) * 255.).to(device=device, dtype=torch.uint8)

    images_batch[:, :, :h, :w] = inputs

    images_batch[:, :, :h, w + pad_size:] = true_masks[:, 3, :, :][:, None, :, :]
    images_batch[:, 0, :h, w + pad_size:] += true_masks[:, 0, :, :]
    images_batch[:, 1, :h, w + pad_size:] += true_masks[:, 1, :, :]
    images_batch[:, 2, :h, w + pad_size:] += true_masks[:, 2, :, :]

    images_batch[:, :, h + pad_size:, :w] = masks_soft[:, 3, :, :][:, None, :, :]
    images_batch[:, 0, h + pad_size:, :w] += masks_soft[:, 0, :, :]
    images_batch[:, 1, h + pad_size:, :w] += masks_soft[:, 1, :, :]
    images_batch[:, 2, h + pad_size:, :w] += masks_soft[:, 2, :, :]

    images_batch[:, :, h + pad_size:, w + pad_size:] = masks_hard[:, 3, :, :][:, None, :, :]
    images_batch[:, 0, h + pad_size:, w + pad_size:] += masks_hard[:, 0, :, :]
    images_batch[:, 1, h + pad_size:, w + pad_size:] += masks_hard[:, 1, :, :]
    images_batch[:, 2, h + pad_size:, w + pad_size:] += masks_hard[:, 2, :, :]
    return images_batch


if __name__ == '__main__':

    if net_name == 'cascadTUNet':
        model = cascadTUNet(n_channels=3, n_classes=5)



    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))


        #####多GPU
        # original saved file with DataParallel
        state_dict = torch.load(args.model)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        #######

        print('Model loaded from {}'.format(args.model))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        sys.exit(0)


    eval_image_paths, eval_mask_paths = get_images(image_dir, args.preprocess, phase='test')


    if net_name == 'cascadTUNet':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, mode='test', augmentation_prob=0)


    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)

    vis_images = eval_model(model, eval_loader)

