import numpy as np
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from torchvision import transforms as T
from torchvision.transforms import functional as F
import random

resize_wh = 800
# transform = transforms.Compose([
#         transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
#         transforms.RandomRotation(rotation_angle),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
# transform_val = transforms.Compose([
#         transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
# mask_transform = transforms.Compose([
#     transforms.ToTensor()
# ])


class IDRIDDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, class_number=0, mode='train',augmentation_prob=0.4):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        if mask_paths is not None:
            self.mask_paths = mask_paths
        self.class_number = class_number
        self.mode = mode
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.image_paths)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path4 = self.mask_paths[idx]
        item = self.pil_loader(image_path)

        info = [item]
        w, h = item.size
        if self.mask_paths is not None:
            for i, mask_path in enumerate(mask_path4):
                if mask_path is None:
                    info.append(Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)))
                else:
                    info.append(self.pil_loader(mask_path))

        image1 = info[0]
        GT1 = info[1]
        GT2 = info[2]
        GT3 = info[3]
        GT4 = info[4]
        Transform = []
        # Transform.append(T.Resize((640, 640)))
        Resize_ = T.Resize((resize_wh, resize_wh))
        # if (self.mode == 'test'):
        #     image1 = Resize_(image1)
        #     GT1 = info[1]
        #     GT2 = info[2]
        #     GT3 = info[3]
        #     GT4 = info[4]
        # else:
        image1 = Resize_(image1)
        GT1 = Resize_(GT1)
        GT2 = Resize_(GT2)
        GT3 = Resize_(GT3)
        GT4 = Resize_(GT4)

        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:

                RotationRange = random.randint(-20, 20)
                Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                Transform = T.Compose(Transform)

                image1 = Transform(image1)
                GT1 = Transform(GT1)
                GT2 = Transform(GT2)
                GT3 = Transform(GT3)
                GT4 = Transform(GT4)

                if random.random() < 0.5:
                    image1 = F.hflip(image1)
                    GT1 = F.hflip(GT1)
                    GT2 = F.hflip(GT2)
                    GT3 = F.hflip(GT3)
                    GT4 = F.hflip(GT4)

                if random.random() < 0.5:
                    image1 = F.vflip(image1)
                    GT1 = F.vflip(GT1)
                    GT2 = F.vflip(GT2)
                    GT3 = F.vflip(GT3)
                    GT4 = F.vflip(GT4)

                Transform = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

                image1 = Transform(image1)
                Transform = []

        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image1 = Transform(image1)
        GT1 = Transform(GT1)
        GT2 = Transform(GT2)
        GT3 = Transform(GT3)
        GT4 = Transform(GT4)

        Norm_ = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image1 = Norm_(image1)

        info_mask = [GT1]
        info_mask.append(GT2)
        info_mask.append(GT3)
        info_mask.append(GT4)

        inputs = np.array(image1)


        # if self.transform:
        #     info = self.transform(info)
        # inputs = np.array(info[0])
        # if inputs.shape[2] == 3:
        #     inputs = np.transpose(np.array(info[0]), (2, 0, 1))
        #     inputs = inputs / 255.

        if len(inmask) > 1:
            masks = np.array([np.array(maskimg)[0, :, :] for maskimg in info_mask[:]])
            masks_sum = np.sum(masks, axis=0)
            empty_mask = 1 - masks_sum
            masks = np.concatenate((empty_mask[None, :, :], masks), axis=0)
            return inputs, masks
        else:
            return inputs


