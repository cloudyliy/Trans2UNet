# CUDA_VISIBLE_DEVICES=0 python eval_npy_unet.py
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
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import matplotlib.pyplot as plt
from unet import UNet

from cascade_transunet_unet import cascadTUNet
import time


from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_curve, auc, f1_score
from res50_unet import Resnet_Unet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = OptionParser()
parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                  type='int', help='batch size')
parser.add_option('-m', '--model', dest='model',
                  default='/home/DR/results/models/unet/model_AP.pth.tar',
                  type='str', help='models stored')
parser.add_option('-n', '--net-name', dest='netname', default='unet',
                  type='str', help='net name, ViT_seg,unet,cascadTUNet')
# parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
#                       default=False, help='preprocess input images')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                  default='1', help='preprocess input images')
# parser.add_option('-i', '--healthy-included', dest='healthyincluded', action='store_true',
#                       default=False, help='include healthy images')
parser.add_option('-v','--vit_name', dest='vit_name',type='str',
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_option('-s','--n_skip', dest='n_skip',type='int', default=3, help='using number of skip-connect, default is num')
parser.add_option('-z','--img_size', dest='img_size',type='int',
                    default=800, help='input patch size of network input')
parser.add_option('-q','--vit_patches_size', dest='vit_patches_size',type='int',
                    default=16, help='vit_patches_size, default is 16')
(args, _) = parser.parse_args()

net_name = args.netname
lesions = ['ex', 'he', 'ma', 'se']

image_dir = 'DR/data'
if net_name == 'unet':
    logdir = 'DR/eval_npy/unet/'
    figure_out_dir = 'DR/figure/unet/'
    args.model = 'results/models/unet/model_AP.pth.tar'


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


            if net_name == 'unet':
                masks_pred = model(inputs).to("cpu")



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

            for i in range(n_number):
                precision, recall, thresholds = precision_recall_curve(true_masks[:, i+1, :, :].reshape(-1), masks_soft[:, i+1, :, :].reshape(-1))

                f1_scores = 2 * recall * precision / (recall + precision)
                best_f1_th = thresholds[np.argmax(f1_scores)]
                mask_predict_avg_binary = torch.where(masks_pred_sigmoid[:, i+1, :, :] > best_f1_th, 1, 0)

                masks_hard[:,i] = mask_predict_avg_binary
            masks_hard = masks_hard.to("cpu")



            np.save(os.path.join(logdir, 'mask_soft_' + str(batch_id) + '.npy'), masks_soft[:, 1:].numpy())

            np.save(os.path.join(logdir, 'mask_true_' + str(batch_id) + '.npy'), true_masks[:, 1:].numpy())

            np.save(os.path.join(logdir, 'mask_hard_' + str(batch_id) + '.npy'), masks_hard[:, :].numpy())
            true_GT = true_masks[:, :].cpu().numpy()

            cv2.imwrite(figure_out_dir_EX + str(batch_id) + '_EX.png', masks_hard[0, 0, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_EX + str(batch_id) + '_soft_EX.png', masks_soft[0, 1, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_EX + str(batch_id) + '_GT_EX.png', true_GT[0, 1, :, :] * 255)

            cv2.imwrite(figure_out_dir_SE + str(batch_id) + '_SE.png', masks_hard[0, 1, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_SE + str(batch_id) + '_soft_SE.png', masks_soft[0, 2, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_SE + str(batch_id) + '_GT_SE.png', true_GT[0, 2, :, :] * 255)

            cv2.imwrite(figure_out_dir_HE + str(batch_id) + '_HE.png', masks_hard[0, 2, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_HE + str(batch_id) + '_soft_HE.png', masks_soft[0, 3, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_HE + str(batch_id) + '_GT_HE.png', true_GT[0, 3, :, :] * 255)

            cv2.imwrite(figure_out_dir_MA + str(batch_id) + '_MA.png', masks_hard[0, 3, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_MA + str(batch_id) + '_soft_MA.png', masks_soft[0, 4, :, :].numpy() * 255)
            cv2.imwrite(figure_out_dir_MA + str(batch_id) + '_GT_MA.png', true_GT[0, 4, :, :] * 255)



            batch_id += 1
    return  vis_images

if __name__ == '__main__':

    if net_name == 'ViT_seg':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=5)

    if net_name == 'unet':
        model = UNet(n_channels=3, n_classes=5)


    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        ##单GPU
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.load_state_dict(checkpoint['g_state_dict'])

        # #####多GPU
        # # original saved file with DataParallel
        # state_dict = torch.load(args.model)
        # # create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        #
        # new_state_dict = OrderedDict()
        # for k, v in state_dict['state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)
        # #######

        print('Model loaded from {}'.format(args.model))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        sys.exit(0)


    eval_image_paths, eval_mask_paths = get_images(image_dir, args.preprocess, phase='test')



    if net_name == 'ViT_seg':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, mode='test', augmentation_prob=0)

    if net_name == 'unet':
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, mode='test', augmentation_prob=0)




    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)

    vis_images = eval_model(model, eval_loader)



