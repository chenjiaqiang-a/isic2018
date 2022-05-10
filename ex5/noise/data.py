from torch.utils import data
from PIL import Image
import os
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd


class NoisyISIC2018(data.Dataset):
    def __init__(self, ann_file: str, img_dir: str, transform=None, target_transform=None,
                 noise_type: str = 'symmetric', noise_rate: float = 0.1, random_state: int = 123):
        """ ISIC 2018 Dataset with noisy labels

        Args:
            ann_file (str): csv annotation file path
            img_dir (str): directory path of images
            transform: input transformation
            target_transform: target transformation
            noise_type (str): noise type ('symmetric', 'asymmetric'). Defaults to 'symmetric'.
            noise_rate (float): rate of noise. Defaults to 0.1.
            random_state (int): random seed. Defaults to 123.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.img_ids, self.clean_labels = self._csv_reader(ann_file)
        self.class_num = len(self.categories)
        self.noisy_labels, self.actual_noise_rate = noisify(
            self.clean_labels, self.noise_type, self.noise_rate, self.class_num, random_state)
        print("Actual noise rate: {:.4f}".format(self.actual_noise_rate))

    def _csv_reader(self, csv_file):
        df = pd.read_csv(csv_file, header=0)
        self.categories = list(df.columns)[1:]
        self.class_dict = {}
        self.label_dict = {}
        for i, name in enumerate(self.categories):
            self.class_dict[name] = i
            self.label_dict[i] = name
        df['label'] = df.select_dtypes(['number']).idxmax(axis=1)
        df['label'] = df['label'].apply(lambda x: self.class_dict[x])
        img_ids = list(df['image'])
        labels = np.array(list(df['label']))
        return img_ids, labels

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, clean_label, noisy_label)
        """

        pth_img = os.path.join(self.img_dir, self.img_ids[idx] + '.jpg')
        img = Image.open(pth_img)
        clean_label = self.clean_labels[idx]
        noisy_label = self.noisy_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            clean_label = self.target_transform(clean_label)
            noisy_label = self.target_transform(noisy_label)

        return img, clean_label, noisy_label

    def __len__(self):
        return len(self.img_ids)

    def to_names(self, nums):
        """ convert a goup of indices to string names 
        
        Args:
            nums(torch.Tensor): a list of number labels

        Return:
            a list of dermatological names
        
        """
        names = [self.label_dict[int(num)] for num in nums]
        return names


def multiclass_noisify(y, P, random_state=123):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.

    Args:
        y (list): a list of index label
        P (matrix): n x n transition matrix with values between [0, 1]
        random_state (int): random seed. Defaults to 123.

    Returns:
        noisy y
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def noisify_symmetric(y, noise_rate, random_state=123, nb_classes=7):
    """ noisify labels in the symmetric way
    """
    # create transition matrix
    P = np.ones((nb_classes, nb_classes))
    ## convert to other classes with equal probabilities (p = noise/(n-1))
    P = (noise_rate / (nb_classes - 1)) * P

    if noise_rate > 0.0:
        for i in range(nb_classes):
            P[i, i] = 1. - noise_rate

        noisy_y = multiclass_noisify(y, P=P, random_state=random_state)
        actual_noise_rate = (noisy_y != y).mean()

    return noisy_y, actual_noise_rate


def noisify_asymmetric(y, noise_rate, random_state=123):
    r""" noisify labels in an asymmetric way: 𝑁𝑉 <-> 𝑀𝐸𝐿, 𝐵𝐶𝐶 <-> 𝐵𝐾𝐿, 𝑉𝐴𝑆𝐶 <-> 𝐷𝐹,
        {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
    """
    P = np.eye(7)
    n = noise_rate

    if n > 0.0:
        # 0 <-> 1
        P[0, 0], P[0, 1] = 1. - n, n
        P[1, 1], P[1, 0] = 1. - n, n

        # 2 <-> 4
        P[2, 2], P[2, 4] = 1. - n, n
        P[4, 4], P[4, 2] = 1. - n, n

        # 5 <-> 6
        P[5, 5], P[5, 6] = 1. - n, n
        P[6, 6], P[6, 5] = 1. - n, n

        # 3 <-> 6
        P[3, 3], P[3, 6] = 1. - n, n

        noisy_y = multiclass_noisify(y, P=P, random_state=random_state)
        actual_noise_rate = (noisy_y != y).mean()

    return noisy_y, actual_noise_rate


def noisify(labels, noise_type='symmetric', noise_rate=0.1, class_num=7, random_state=123):
    assert noise_rate >= 0 and noise_rate <= 1, "Noise rate is not in [0, 1]"
    if noise_rate == 0:
        return labels, 0
    if noise_type == 'symmetric':
        noisy_labels, actual_noise_rate = noisify_symmetric(
            labels, noise_rate, random_state=random_state, nb_classes=class_num)
    elif noise_type == 'asymmetric':
        noisy_labels, actual_noise_rate = noisify_asymmetric(
            labels, noise_rate, random_state=random_state)
    else:
        raise ValueError('Not Implemented')
    return noisy_labels, actual_noise_rate
