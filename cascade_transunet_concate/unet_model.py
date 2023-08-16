# full assembly of the sub-parts to form the complete net

import torch
from .unet_parts import *
import torch.nn as nn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-v','--vit_name', dest='vit_name',type='str',
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_option('-s','--n_skip', dest='n_skip',type='int', default=3, help='using number of skip-connect, default is num')
parser.add_option('-z','--img_size', dest='img_size',type='int',
                    default=800, help='input patch size of network input')
parser.add_option('-q','--vit_patches_size', dest='vit_patches_size',type='int',
                    default=16, help='vit_patches_size, default is 16')

(args, _) = parser.parse_args()

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outc(x)
        return x

class cascadTUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(cascadTUNet, self).__init__()
        # self.unet1 = UNet(n_channels, n_classes)
        # self.tunet =ViT_seg(config_vit, img_size=args.img_size, num_classes=5)
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        self.tunet = ViT_seg(config_vit, img_size=args.img_size, num_classes=5)

        self.unet2 = UNet(n_channels+1, n_classes)
        self.unet3 = UNet(n_channels+1, n_classes)
        self.unet4 = UNet(n_channels+1, n_classes)
        self.unet5 = UNet(n_channels+1, n_classes)
        self.conv = nn.Conv2d(n_classes, n_classes, 1)

        self.conv_c0 = nn.Conv2d(4, 1, 1)
        self.conv_c1 = nn.Conv2d(4, 1, 1)
        self.conv_c2 = nn.Conv2d(4, 1, 1)
        self.conv_c3 = nn.Conv2d(4, 1, 1)
        self.conv_c4 = nn.Conv2d(4, 1, 1)


    def forward(self, x):
        x1 = self.tunet(x)
        x1_0 = torch.unsqueeze(x1[:, 0, :, :],1)
        x1_1 = torch.unsqueeze(x1[:, 1, :, :],1)
        x1_2 = torch.unsqueeze(x1[:, 2, :, :],1)
        x1_3 = torch.unsqueeze(x1[:, 3, :, :],1)
        x1_4 = torch.unsqueeze(x1[:, 4, :, :],1)

        # print("x ",x.shape)
        # print("x1 ", x1.shape)
        # print("x1_1 ", x1_1.shape)
        x_1 = torch.cat([x, x1_1], dim=1)

        x2 = self.unet2(x_1)
        x_2 = torch.cat([x, x1_2], dim=1)

        x3 = self.unet3(x_2)
        x_3 = torch.cat([x, x1_3], dim=1)

        x4 = self.unet4(x_3)
        x_4 = torch.cat([x, x1_4], dim=1)

        x5 = self.unet5(x_4)

        c_0 = torch.stack((x2[:,0,:,:],x3[:,0,:,:],x4[:,0,:,:],x5[:,0,:,:]),1)  ####H*W*5 ,d1[:,0,:,:].shape [8, 512, 512]
        # print('c0',c_0.shape)
        c_1 = torch.stack((x2[:,1,:,:],x3[:,1,:,:],x4[:,1,:,:],x5[:,1,:,:]),1)   ####EX
        c_2 = torch.stack((x2[:, 2, :, :], x3[:, 2, :, :], x4[:,2, :, :], x5[:, 2, :, :]), 1)   ###SE
        c_3 = torch.stack((x2[:, 3, :, :], x3[:, 3, :, :], x4[:, 3, :, :], x5[:,3, :, :]), 1)   ###HE
        c_4 = torch.stack((x2[:, 4, :, :], x3[:, 4, :, :], x4[:, 4, :, :], x5[:, 4, :, :]), 1)  ####MA

        c0 = self.conv_c0(c_0) ### H*W*1
        c1 = self.conv_c1(c_1)
        c2 = self.conv_c2(c_2)
        c3 = self.conv_c3(c_3)
        c4 = self.conv_c4(c_4)
        x6 = torch.cat((c1, c2, c3, c4), 1)

        return x1,x2,x3,x4,x5,x6