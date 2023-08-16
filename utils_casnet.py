#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
from preprocess import clahe_gridsize
import cv2

import torch.nn as nn


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


train_ratio = 0.8
eval_ratio = 0.2


def get_images(image_dir, preprocess='0', phase='train'):
    if phase == 'train' or phase == 'eval':
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet'

    imgs = glob.glob(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, '*.jpg'))

    imgs.sort()
    mask_paths = []
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * eval_ratio)
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs

    mask_path = os.path.join(image_dir, 'Groundtruths', setname)
    lesions = ['HardExudates', 'SoftExudates', 'Haemorrhages', 'Microaneurysms']
    lesion_abbvs = ['EX', 'SE', 'HE', 'MA']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion, lesion_abbv in zip(lesions, lesion_abbvs):
            candidate_path = os.path.join(mask_path, lesion, name + '_' + lesion_abbv + '.tif')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    return image_paths, mask_paths

