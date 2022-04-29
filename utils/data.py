import os
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils import data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ISIC2018Dataset(data.Dataset):
    """ ISIC 2018 Dataset

    Args:
        csv_path(str): csv file path of ISIC 2018.
        img_path(str): image folder of ISIC 2018.
        transform: image transform option.
    """
    def __init__(self, csv_file_path: str, img_dir: str, transform=None, target_transform=None, **kwargs):
        super(ISIC2018Dataset, self).__init__(**kwargs)

        self.img_dir = img_dir
        self.trans = transform
        self.target_trans = target_transform

        df = pd.read_csv(csv_file_path)
        self.target_to_label = list(df.columns.values())[1:]
        self.label_to_target = {
            label: target for target, label in enumerate(self.target_to_label)
        }
        arr = np.array(df[self.target_to_label])

        self.img_names = list(df['image'])
        self.targets = list(arr.argmax(axis=1))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index] + '.jpg')
        image = Image.open(img_path)
        target = self.targets[index]

        if self.trans is not None:
            image = self.trans(image)
        if self.target_trans is not None:
            target = self.target_trans(target)

        return image, target

    def count_samples(self) -> list:
        """ count sample_nums """
        counter = Counter(self.targets)
        class_nums = [(self.target_to_label[target], num) for target, num in counter.items()]
        return class_nums

    def to_targets(self, label_list: list) -> list:
        targets = [self.label_to_target[i] for i in label_list]
        return targets

    def to_labels(self, target_list: list) -> list:
        labels = [self.target_to_label[int(i)] for i in target_list]
        return labels
