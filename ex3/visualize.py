# visualization
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import cv2


######################### display samples ########################
def imshow(image, label, ax=None, normalize=True):
    """show single along with label on an ax"""
    
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(label)

    return ax


def show_samples(images, labels, nrows=2, ncols=3, title=None, normalize=True, dpi=100):
    """ show multiple samples

    args:
        nrows (int, optional): number of row
        ncols (int, optional): number of column
        title (str, optional): title.
        normalize (bool, optional): whether the images are normalized
        dpi (int): dpi for plotting
    """
    fig, axes = plt.subplots(nrows, ncols, facecolor='#ffffff', dpi=dpi)

    if nrows * ncols == 1:
        axes = imshow(images, labels, axes, normalize)
    else:
        # .flat: to map samples to multi-dimensional axes
        for (ax, image, label) in zip(axes.flat, images, labels):
            ax = imshow(image, label, ax, normalize)

    fig.suptitle(title)
    fig.tight_layout = True
    fig.subplots_adjust(top=0.85, hspace=0.3)
    plt.show()
    
    
def segshow(image, mask, ax=None, normalize=True):
    """show single along with label on an ax"""
    
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((1, 2, 0))
        mask = mask.numpy() # .transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.imshow(mask, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def show_seg_samples(images, masks, nrows=2, ncols=3, title=None, normalize=True, dpi=100):
    """ show multiple samples

    args:
        nrows (int, optional): number of row
        ncols (int, optional): number of column
        title (str, optional): title.
        normalize (bool, optional): whether the images are normalized
        dpi (int): dpi for plotting
    """
    fig, axes = plt.subplots(nrows, ncols, facecolor='#ffffff', dpi=dpi)

    if nrows * ncols == 1:
        axes = imshow(images, labels, axes, normalize)
    else:
        # .flat: to map samples to multi-dimensional axes
        for (ax, image, mask) in zip(axes.flat, images, masks):
            ax = segshow(image, mask, ax, normalize)

    fig.suptitle(title)
    fig.tight_layout = True
    fig.subplots_adjust(top=0.85, hspace=0.3)
    plt.show()
    
    
def show_predictions(images, labels, preds, nrows=2, ncols=3, title=None, normalize=True, dpi=100):
    """ show multiple samples along with predictions and labels

    return:
        images with title: pred[true]
    """
    fig, axes = plt.subplots(nrows, ncols, facecolor='#ffffff', dpi=dpi)

    if nrows * ncols == 1:
        axes = imshow(images, '{}[{}]'.format(preds, labels), axes, normalize)
    else:
        # .flat: to map samples to multi-dimensional axes
        for (ax, image, label, pred) in zip(axes.flat, images, labels, preds):
            ax = imshow(image, '{}[{}]'.format(pred, label), ax, normalize)

    fig.suptitle(title)
    fig.tight_layout = True
    fig.subplots_adjust(top=0.85, hspace=0.3)
    plt.show()


################################# evaluation ##############################

def draw_confusion(cf_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    # based on specificity
    sns.heatmap(cf_matrix/np.sum(cf_matrix), ax=axes[0], annot=True, fmt='.2%', cmap='Blues', annot_kws={"size":8}, cbar=True)
    # based on sensitivity
    sns.heatmap(cf_matrix/np.sum(np.array(cf_matrix), axis=1, keepdims=True), ax=axes[1], annot=True, fmt='.2%', cmap='Blues', annot_kws={"size":8})
    for ax, title in zip(axes, ['specificity', 'sensitivity']):
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)


# draw the CAM

def draw_cam(ax, cam):
    """ draw the CAM
    Args:
        ax: plt的画轴
        cam(2d ndarray): the cam matrix
    """
    image = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.axis("off")
    ax.imshow(image)