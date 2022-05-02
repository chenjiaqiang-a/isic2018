from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as m_colors
import cv2

__all__ = ["draw_image", "draw_samples", "plot_confusion_matrix",
           "plot_roc_curves", "plot_losses", "draw_cam"]


# display samples
def draw_image(image, label, ax=None):
    """show single along with label on an ax"""
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(label)
    return ax


def draw_samples(images, labels, nrows=2, ncols=3, title=None):
    """ show multiple samples

    args:
        nrows (int, optional): number of row
        ncols (int, optional): number of column
        title (str, optional): title.
        dpi (int): dpi for plotting
    """
    fig, axes = plt.subplots(nrows, ncols, facecolor='w', dpi=100)

    for (ax, image, label) in zip(axes.flat, images, labels):
        ax = draw_image(image, label, ax)

    fig.suptitle(title)
    fig.tight_layout = True
    fig.subplots_adjust(top=0.85, hspace=0.2)
    return fig, axes


# evaluation
def plot_confusion_matrix(cm, classes, title=None, filename="cm.png", cmap="Blues"):
    plt.rc('font', size='8')  # 设置字体大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    plt.figure(figsize=(10, 10), facecolor='w')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_roc_curves(fpr, tpr, roc_auc, filename="cm.png", categories=None):
    # 绘制全部的ROC曲线
    n_classes = len(fpr) - 2

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(list(m_colors.TABLEAU_COLORS.keys()))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=m_colors.TABLEAU_COLORS[color], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(categories[i] if categories else i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()


def plot_losses(losses, title="", legend=None, filename="losses.png"):
    plt.figure()
    for i, l in enumerate(losses):
        plt.plot(l, label=legend[i] if legend else None)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.show()


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
