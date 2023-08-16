# CUDA_VISIBLE_DEVICES=0,1 python train_trans2unet.py
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

from cascade_transunet_concate import cascadTUNet

from utils_casnet import get_images


from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import copy
# from logger import Logger
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score,confusion_matrix,roc_curve, auc,f1_score
import random


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(10)

parser = OptionParser()
parser.add_option('-e', '--epochs', dest='epochs', default=600, type='int',
                  help='number of epochs')
parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                  type='int', help='batch size')
parser.add_option('-l', '--learning-rate', dest='lr', default=0.0002,  #0.0002
                  type='float', help='learning rate')
parser.add_option('-r', '--resume', dest='resume',
                  default=False, help='resume file model')
parser.add_option('-p', '--log-dir', dest='logdir', default='drlog',
                  type='str', help='tensorboard log')
parser.add_option('-m', '--model-dir', dest='modeldir', default='./models',
                  type='str', help='models stored')
parser.add_option('-n', '--net-name', dest='netname', default='cascadTUNet',
                  type='str', help='net name, unet or hednet or cascadTUNet')
# parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
#                       default=False, help='preprocess input images')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                  default='1', help='preprocess input images')
parser.add_option('-i', '--healthy-included', dest='healthyincluded', action='store_true',
                  default=False, help='include healthy images')
parser.add_option('-a', '--active-learning', dest='al', action='store_true',
                  default=False, help='whether to use active learning')

(args, _) = parser.parse_args()

# logger = Logger('./logs', args.logdir)
dir_checkpoint = args.modeldir
net_name = args.netname
lesion_dice_weights = [0., 0., 0., 0.]
lesions = ['ex', 'he', 'ma', 'se']

image_dir_train = 'DR/data'


softmax = nn.Softmax(1)


def eval_model(model, eval_loader,  criterion,criterion_EX1,criterion_SE1,criterion_HE1,criterion_MA1, criterion_EX2,criterion_SE2,criterion_HE2,criterion_MA2,):
    model.eval()
    eval_tot = len(eval_loader)
    eval_loss_ce = 0.
    ap = 0.

    soft_masks_all = []
    true_masks_all = []
    AP = []

    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:

            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            if net_name == 'unet':
                masks_pred = model(inputs)

            elif net_name == 'cascadTUNet':
                x1,x2,x3,x4,x5,x6 = model(inputs)
                masks_pred = x6

            masks_pred_transpose = x1.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_indices = torch.argmax(true_masks, 1)
            true_masks_flat = true_masks_indices.reshape(-1)
            loss_ce = criterion(masks_pred_flat, true_masks_flat.long())

            masks_pred_transpose2 = x2.permute(0, 2, 3, 1)
            masks_pred_flat2 = masks_pred_transpose2.reshape(-1, masks_pred_transpose2.shape[-1])
            loss_ce1 = criterion_EX1(masks_pred_flat2, true_masks_flat.long())

            masks_pred_transpose3 = x3.permute(0, 2, 3, 1)
            masks_pred_flat3 = masks_pred_transpose3.reshape(-1, masks_pred_transpose3.shape[-1])
            loss_ce2 = criterion_SE1(masks_pred_flat3, true_masks_flat.long())

            masks_pred_transpose4 = x4.permute(0, 2, 3, 1)
            masks_pred_flat4 = masks_pred_transpose4.reshape(-1, masks_pred_transpose4.shape[-1])
            loss_ce3 = criterion_HE1(masks_pred_flat4, true_masks_flat.long())

            masks_pred_transpose5 = x5.permute(0, 2, 3, 1)
            masks_pred_flat5 = masks_pred_transpose5.reshape(-1, masks_pred_transpose5.shape[-1])
            loss_ce4 = criterion_MA1(masks_pred_flat5, true_masks_flat.long())


            masks_pred2 = x6[:, 0, :, :]

            true_masks2 = true_masks[:, 1, :, :]
            masks_pred_flat2 = masks_pred2.reshape(-1)
            true_masks_flat2 = true_masks2.reshape(-1)
            loss_1 = criterion_EX2(masks_pred_flat2, true_masks_flat2)

            masks_pred3 = x6[:, 1, :, :]
            true_masks3 = true_masks[:, 2, :, :]
            masks_pred_flat3 = masks_pred3.reshape(-1)
            true_masks_flat3 = true_masks3.reshape(-1)
            loss_2 = criterion_SE2(masks_pred_flat3, true_masks_flat3)

            masks_pred4 = x6[:, 2, :, :]
            true_masks4 = true_masks[:, 3, :, :]
            masks_pred_flat4 = masks_pred4.reshape(-1)
            true_masks_flat4 = true_masks4.reshape(-1)
            loss_3 = criterion_HE2(masks_pred_flat4, true_masks_flat4)

            masks_pred5 = x6[:, 3, :, :]
            true_masks5 = true_masks[:, 4, :, :]
            masks_pred_flat5 = masks_pred5.reshape(-1)
            true_masks_flat5 = true_masks5.reshape(-1)
            loss_4 = criterion_MA2(masks_pred_flat5, true_masks_flat5)

            eval_loss_ce += (loss_ce+loss_ce1+loss_ce2+loss_ce3+loss_ce4+loss_1+loss_2+loss_3+loss_4)/9

            masks_pred_softmax = torch.sigmoid(masks_pred)


####AP验证
            n_number = 4
            masks_soft = masks_pred_softmax[:, :, :, :].cpu().numpy()


            true_masks = torch.where(true_masks[:, 1:, :, :] > 0.5, 1,0)
            masks_true = true_masks.cpu().numpy()


            soft_masks_all.extend(masks_soft)
            true_masks_all.extend(masks_true)

        soft_masks_all = np.array(soft_masks_all)
        true_masks_all = np.array(true_masks_all)
        predicted = np.transpose(soft_masks_all, (1, 0, 2, 3)) ##(1, 0, 2, 3)  (2, 0, 1, 3, 4)
        predicted = predicted.round(2)
        gt = np.transpose(true_masks_all, (1, 0, 2, 3))
        predicted = np.reshape(predicted, (predicted.shape[0], -1))
        gt = np.reshape(gt, (gt.shape[0], -1))

        for i in range(n_number):
            ap += average_precision_score(gt[i], predicted[i])


        return  eval_loss_ce / eval_tot, ap / n_number


def denormalize(inputs):
    if net_name == 'unet':
        return (inputs * 255.).to(device=device, dtype=torch.uint8)
    else:
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        return ((inputs * std[None, :, None, None] + mean[None, :, None, None]) * 255.).to(device=device,
                                                                                           dtype=torch.uint8)


def generate_log_images(inputs_t, true_masks_t, masks_pred_softmax_t):
    true_masks = (true_masks_t * 255.).to(device=device, dtype=torch.uint8)
    masks_pred_softmax = (masks_pred_softmax_t.detach() * 255.).to(device=device, dtype=torch.uint8)
    inputs = denormalize(inputs_t)
    bs, _, h, w = inputs.shape
    pad_size = 5
    images_batch = (torch.ones((bs, 3, h * 3 + pad_size * 2, w * 4 + pad_size * 3)) * 255.).to(device=device,
                                                                                               dtype=torch.uint8)

    images_batch[:, :, :h, :w] = inputs

    images_batch[:, :, h + pad_size:h * 2 + pad_size, :w] = 0
    images_batch[:, 0, h + pad_size:h * 2 + pad_size, :w] = true_masks[:, 1, :, :]

    images_batch[:, :, h + pad_size:h * 2 + pad_size, w + pad_size:w * 2 + pad_size] = 0
    images_batch[:, 1, h + pad_size:h * 2 + pad_size, w + pad_size:w * 2 + pad_size] = true_masks[:, 2, :, :]

    images_batch[:, :, h + pad_size:h * 2 + pad_size, w * 2 + pad_size * 2:w * 3 + pad_size * 2] = 0
    images_batch[:, 2, h + pad_size:h * 2 + pad_size, w * 2 + pad_size * 2:w * 3 + pad_size * 2] = true_masks[:, 3, :,
                                                                                                   :]

    images_batch[:, :, h + pad_size:h * 2 + pad_size, w * 3 + pad_size * 3:] = true_masks[:, 4, :, :][:, None, :, :]

    images_batch[:, :, h * 2 + pad_size * 2:, :w] = 0
    images_batch[:, 0, h * 2 + pad_size * 2:, :w] = masks_pred_softmax[:, 1, :, :]

    images_batch[:, :, h * 2 + pad_size * 2:, w + pad_size:w * 2 + pad_size] = 0
    images_batch[:, 1, h * 2 + pad_size * 2:, w + pad_size:w * 2 + pad_size] = masks_pred_softmax[:, 2, :, :]

    images_batch[:, :, h * 2 + pad_size * 2:, w * 2 + pad_size * 2:w * 3 + pad_size * 2] = 0
    images_batch[:, 2, h * 2 + pad_size * 2:, w * 2 + pad_size * 2:w * 3 + pad_size * 2] = masks_pred_softmax[:, 3, :,
                                                                                           :]

    images_batch[:, :, h * 2 + pad_size * 2:, w * 3 + pad_size * 3:] = masks_pred_softmax[:, 4, :, :][:, None, :, :]

    return images_batch


def train_model(model, train_loader, eval_loader,  criterion,criterion_EX1,criterion_SE1,criterion_HE1,criterion_MA1, criterion_EX2,criterion_SE2,criterion_HE2,criterion_MA2, optimizer, scheduler, batch_size, num_epochs=5,
                start_epoch=0, start_step=0):
    ###多个GPU的时候加上
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids = [0, 1])
    model.to(device=device)
    tot_step_count = start_step

    best_ap = 0.
    best_dice = 0.
    # dir_checkpoint = 'results/models'
    dir_checkpoint = 'results/models/Trans2UNet/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)


    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, start_epoch + num_epochs))
        # scheduler.step()
        model.train()
        loss_sum = 0
        epoch_loss_ce = 0
        N_train = len(train_dataset)
        batch_step_count = 0
        vis_images = []
        for inputs, true_masks in tqdm(train_loader):

            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)


            if net_name == 'unet':
                masks_pred = model(inputs)
            elif net_name == 'cascadTUNet':
                x1, x2, x3, x4, x5, x6 = model(inputs)


            masks_pred_transpose = x1.permute(0, 2, 3, 1)
            masks_pred_flat = masks_pred_transpose.reshape(-1, masks_pred_transpose.shape[-1])
            true_masks_indices = torch.argmax(true_masks, 1)
            true_masks_flat = true_masks_indices.reshape(-1)
            loss_ce = criterion(masks_pred_flat, true_masks_flat.long())

            masks_pred_transpose2 = x2.permute(0, 2, 3, 1)
            masks_pred_flat2 = masks_pred_transpose2.reshape(-1, masks_pred_transpose2.shape[-1])
            loss_ce1 = criterion_EX1(masks_pred_flat2, true_masks_flat.long())

            masks_pred_transpose3 = x3.permute(0, 2, 3, 1)
            masks_pred_flat3 = masks_pred_transpose3.reshape(-1, masks_pred_transpose3.shape[-1])
            loss_ce2 = criterion_SE1(masks_pred_flat3, true_masks_flat.long())

            masks_pred_transpose4 = x4.permute(0, 2, 3, 1)
            masks_pred_flat4 = masks_pred_transpose4.reshape(-1, masks_pred_transpose4.shape[-1])
            loss_ce3 = criterion_HE1(masks_pred_flat4, true_masks_flat.long())

            masks_pred_transpose5 = x5.permute(0, 2, 3, 1)
            masks_pred_flat5 = masks_pred_transpose5.reshape(-1, masks_pred_transpose5.shape[-1])
            loss_ce4 = criterion_MA1(masks_pred_flat5, true_masks_flat.long())


            masks_pred2 = x6[:, 0, :, :]
            true_masks2 = true_masks[:, 1, :, :]
            masks_pred_flat2 = masks_pred2.reshape(-1)
            true_masks_flat2 = true_masks2.reshape(-1)
            loss_1 = criterion_EX2(masks_pred_flat2, true_masks_flat2)


            masks_pred3 = x6[:, 1, :, :]
            true_masks3 = true_masks[:, 2, :, :]
            masks_pred_flat3 = masks_pred3.reshape(-1)
            true_masks_flat3 = true_masks3.reshape(-1)
            loss_2 = criterion_SE2(masks_pred_flat3, true_masks_flat3)

            masks_pred4 = x6[:, 2, :, :]
            true_masks4 = true_masks[:, 3, :, :]
            masks_pred_flat4 = masks_pred4.reshape(-1)
            true_masks_flat4 = true_masks4.reshape(-1)
            loss_3 = criterion_HE2(masks_pred_flat4, true_masks_flat4)


            masks_pred5 = x6[:, 3, :, :]
            true_masks5 = true_masks[:, 4, :, :]
            masks_pred_flat5 = masks_pred5.reshape(-1)
            true_masks_flat5 = true_masks5.reshape(-1)
            loss_4 = criterion_MA2(masks_pred_flat5, true_masks_flat5)


            ce_weight = 1.
            loss = (loss_ce+loss_ce1+loss_ce2+loss_ce3+loss_ce4+loss_1+loss_2+loss_3+loss_4)/9


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss

            batch_step_count += 1
            tot_step_count += 1

        scheduler.step()
        num = len(train_loader)
        loss_ave = loss_sum / num
        print("loss_train:{}".format(loss_ave))


        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)

        if (epoch + 1) % 1 == 0:
            loss_val,ap = eval_model(model, eval_loader,  criterion,criterion_EX1,criterion_SE1,criterion_HE1,criterion_MA1, criterion_EX2,criterion_SE2,criterion_HE2,criterion_MA2)
            print("loss_val:{}  ".format(loss_val))
            print("AP_val:{}".format(ap))

            if ap > best_ap:
                best_ap = ap
                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    # 'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(state, \
                           os.path.join(dir_checkpoint, 'model_AP.pth.tar'))


if __name__ == '__main__':

    if net_name == 'cascadTUNet':
        model = cascadTUNet(n_channels=3, n_classes=5)
    # if net_name == 'UNet':
    #     model = UNet(n_channels=3, n_classes=5)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            print('Model loaded from {}'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
        start_step = 0


    train_image_paths, train_mask_paths = get_images(image_dir_train, args.preprocess, phase='train')
    eval_image_paths, eval_mask_paths = get_images(image_dir_train, args.preprocess, phase='eval')

    augmentation_prob = random.random()

    if net_name == 'cascadTUNet':
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 4, mode='train',augmentation_prob=0.6)
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, mode='val',augmentation_prob=0)
    if net_name == 'UNet':
        train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 4, mode='train',augmentation_prob=0.6)
        eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, mode='val',augmentation_prob=0)

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1., 2., 1., 2.]).to(device))   ###['ex', 'se', 'he', 'ma']

    criterion_EX1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 2., 2., 1., 2.]).to(device))   ###loss_EX
    criterion_SE1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1., 4., 1., 2.]).to(device))   ###loss_SE
    criterion_HE1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1., 2., 2., 2.]).to(device))   ###loss_HE
    criterion_MA1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 1., 2., 1., 4.]).to(device))  ###loss_MA


    criterion_EX2 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.]).to(device))
    criterion_SE2 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([40.]).to(device))
    criterion_HE2 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.]).to(device))
    criterion_MA2 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([40.]).to(device))


    train_model(model, train_loader, eval_loader, criterion,criterion_EX1,criterion_SE1,criterion_HE1,criterion_MA1, criterion_EX2,criterion_SE2,criterion_HE2,criterion_MA2,optimizer, scheduler, args.batchsize,
                num_epochs=args.epochs, start_epoch=start_epoch, start_step=start_step)
