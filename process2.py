import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

INPUT_DIR = 'Images_CLAHE0/TestingSet/'
INPUT_MASK_DIR = 'IDRiD_mask/TestingSet_v/'
OUTPUT_DIR = 'Images_CLAHE5/TestingSet/'


list_names = glob.glob(INPUT_DIR + '*.jpg')
list_name_f = [x.split('\\')[-1].split('.')[0] for x in list_names]
print(len(list_names))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


# def load_ben_color(path, sigmaX=10 ):
#     image = cv2.imread(path)
#     image = crop_image_from_gray(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = cv2.resize(image,(640,640))
#     # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     height, width, depth = image.shape
#
#     x = int(width / 2)
#     y = int(height / 2)
#     r = np.amin((x, y))
#
#     circle_img = np.zeros((height, width), np.uint8)
#     cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
#     img = cv2.bitwise_and(image, image, mask=circle_img)
#     image = crop_image_from_gray(img)
#
#     image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
#     return image


def load_ben_color(path, sigmaX=10 ):
    image = cv2.imread(path)
    # image = crop_image_from_gray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # image = crop_image_from_gray(image)

    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image


for file_name_temp in tqdm(list_name_f):
    path = INPUT_DIR + file_name_temp + '.jpg'
    img = cv2.imread(path)
    h,w,_ = img.shape
    # print(img.shape)
    X = 800/30
    image = load_ben_color(path, sigmaX=25)#25
    mask = cv2.imread(INPUT_MASK_DIR + file_name_temp + '_MASK.tif')
    print('image:', file_name_temp)
    mask = mask/255
    image = image*mask

    # print(image.shape)
    cv2.imwrite(OUTPUT_DIR + file_name_temp + '.jpg', image)

