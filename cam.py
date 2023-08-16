import argparse
import cv2
import numpy as np
import scipy.io as sio
import torch
from torchvision import models
from resnet import ResNet32
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def build_model():
    model = ResNet32(num_classes=2)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = False

    return model


# create model
model = build_model()
print(model)

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    # model = models.resnet50(pretrained=True)
    model.load_state_dict(torch.load('/home/tabgha/users/fs/Project/AdversarialExamples/06-Result/model/Resnet/Resnet_epoch100.pth'))

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.layer3[-1]
    print(target_layer)

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)
    # load data 
    imgfile = sio.loadmat('/home/tabgha/users/fs/Project/AdversarialExamples/01-DataEXP/mat_demo/test/1/img_PGD.mat')
    img = imgfile['img']   # (224, 224)
    # img[img < -900] = -900
    # img[img > -400] = -400
    # img = (img + 900) / 500  # 窗宽[-900,-400]，归一化操作
    rgb_img = np.array([img,img,img])
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    img = img[np.newaxis, np.newaxis, :, :]
    input_tensor = torch.from_numpy(img)
    print(rgb_img.shape)
    # print( input_tensor.shape)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam_PGD.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb_PGD.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb_PGD.jpg', cam_gb)
