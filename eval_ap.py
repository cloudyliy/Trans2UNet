# python eval_ap.py
import os
from optparse import OptionParser
from sklearn.metrics import precision_recall_curve, average_precision_score,confusion_matrix,roc_curve, auc,f1_score
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import type_of_target

parser = OptionParser()
parser.add_option('-p', '--log-dir', dest='logdir', default='eval',
                    type='str', help='tensorboard log')
(args, _) = parser.parse_args()

titles = ['ex', 'se', 'he', 'ma']

# logdir = args.logdir

logdir = '/home/eval_npy/Trans2unet'


if not os.path.exists(logdir):
    os.mkdir(logdir)

def plot_precision_recall_all(predicted, predicted_hard,gt):
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []

    n_number = 4
    for i in range(n_number):
        print(type_of_target(gt[i]))
        precision, recall, _ = precision_recall_curve(gt[i], predicted[i])
        l, = plt.plot(recall, precision, color=colors[i], lw=2)
        # ap = average_precision_score(gt[i], predicted[i])
        ap = auc(recall, precision)
        lines.append(l)
        labels.append('Precision-recall for {}: AP = {:.4f}'.format(titles[i], ap))

        fpr, tpr, threshold = roc_curve(gt[i], predicted[i])
        roc_auc = auc(fpr, tpr)


        print('results for {}: AP {:.4f} AUC {:.4f}'.format(titles[i], ap,roc_auc))


if __name__ == '__main__':
    soft_npy_paths = glob.glob(os.path.join(logdir, '*soft*.npy'))
    hard_npy_paths = glob.glob(os.path.join(logdir, '*hard*.npy'))
    true_npy_paths = glob.glob(os.path.join(logdir, '*true*.npy'))
    soft_npy_paths.sort()
    hard_npy_paths.sort()
    true_npy_paths.sort()
    soft_masks_all = []
    hard_masks_all = []
    true_masks_all = []
    for soft_npy_path, hard_npy_path,true_npy_path in zip(soft_npy_paths, hard_npy_paths,true_npy_paths):
        soft_masks = np.load(soft_npy_path)
        hard_masks = np.load(hard_npy_path)
        true_masks = np.load(true_npy_path)
        print('soft_masks:', soft_masks.shape)
        print('hard_masks:', hard_masks.shape)
        print('true_masks:', true_masks.shape)
        soft_masks_all.append(soft_masks)
        hard_masks_all.append(hard_masks)
        true_masks_all.append(true_masks)
    soft_masks_all = np.array(soft_masks_all)
    hard_masks_all = np.array(hard_masks_all)
    true_masks_all = np.array(true_masks_all)
    print('soft_shape:',soft_masks_all.ndim)
    print('hard_shape:',hard_masks_all.ndim)
    print('true_shape:',true_masks_all.ndim)
    
    predicted = np.transpose(soft_masks_all, (2, 0, 1, 3, 4))
    predicted = predicted.round(2)

    predicted_hard = np.transpose(hard_masks_all, (2, 0, 1, 3, 4))

    gt = np.transpose(true_masks_all, (2, 0, 1, 3, 4))
    predicted = np.reshape(predicted, (predicted.shape[0], -1))
    print('predicted.size', predicted.shape)
    predicted_hard = np.reshape(predicted_hard, (predicted_hard.shape[0], -1)).astype(int)
    gt = np.reshape(gt, (gt.shape[0], -1))
    aps = []
    plot_precision_recall_all(predicted, predicted_hard,gt)
