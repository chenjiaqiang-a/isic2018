import pandas as pd
from torch.utils import data
from PIL import Image, ImageFile
import os
import random
from torchvision import transforms
import albumentations as A  # image augmentation library for segmentation tasks, https://albumentations.ai/docs/
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Annotation(object):
    """ annotate ISIC 2017

    Attributes:
        df(pd.DataFrame): df.columns=['image_id', 'label']
        categories(list): dermatological types
        class_dict(dict): class name -> index
        label_dict(dict): index -> class name
        class_num(int): the number of classes
        
    Usages:
        count_samples(): get numbers of samples in each class

    """
    def __init__(self, ann_file: str) -> None:
        """
        Args:
            ann_file (str): csv file path
        """
        self.df = pd.read_csv(ann_file, header=0)
        self.df['benign'] = 1 - self.df.select_dtypes(['number']).sum(axis=1)
        self.categories = list(self.df.columns)
        self.categories.pop(0)
        self.class_num = len(self.categories)
        self.class_dict, self.label_dict = self._make_dicts()
        self.df = self._relabel()
        # self.class_nums = self.count_samples()

    def _make_dicts(self):
        """ make class and label dict from categories' names """
        class_dict = {}
        label_dict = {}
        for i, name in enumerate(self.categories):
            class_dict[name] = i
            label_dict[i] = name

        return class_dict, label_dict

    def _relabel(self) -> pd.DataFrame:
        self.df['label'] = self.df.select_dtypes(['number']).idxmax(axis=1)
        self.df['label'] = self.df['label'].apply(lambda x: self.class_dict[x])
        for name in self.categories:
            del self.df[name]
        return self.df

    def count_samples(self) -> list:
        """ count sample_nums """
        value_counts = self.df.iloc[:, 1].value_counts()
        class_nums = [value_counts[i] for i in range(len(value_counts))]
        return class_nums

    def to_names(self, nums):
        """ convert a goup of indices to string names 
        
        Args:
            nums(torch.Tensor): a list of number labels

        Return:
            a list of dermatological names
        
        """
        names = [self.label_dict[int(num)] for num in nums]
        return names


class Data(data.Dataset):
    def __init__(self, annotations: pd.DataFrame, img_dir: str, patch_num, transform=None, target_transform=None):
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.patch_num = patch_num  

    def __len__(self):
        return len(self.img_labels) * self.patch_num

    def __getitem__(self, idx: int):
        if self.patch_num == 1:
            # temp for test
            idx_sample = idx
            img_path = os.path.join(self.img_dir, self.img_labels.image_id[idx_sample] + '.jpg')
        else:
            idx_sample = idx // self.patch_num     # index of the original image
            idx_patch = idx % self.patch_num       # index of the patch
            img_path = os.path.join(self.img_dir, self.img_labels.image_id[idx_sample]+ '_' + '{:02d}'.format(idx_patch) + '.jpg')
        image = Image.open(img_path)
        target = self.img_labels['label'].iloc[idx_sample]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class RandomPatch(data.Dataset):
    def __init__(self, annotations, img_dir: str, transform=None, target_transform=None):
        self.img_labels = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.scales = [1/5, 2/5, 3/5, 4/5, 2/5, 3/5, 4/5, 3/5, 4/5]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.image_id[idx] + '.jpg')
        image = Image.open(img_path)
        image = self.rescale_crop(image)
        target = self.img_labels['label'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def rescale_crop(self, image):
        scale = self.scales[random.randint(0, 8)]
        w, h = image.size
        if scale > 1/2:
            trans = transforms.Compose([
                transforms.RandomCrop((int(h * scale), int(w * scale)), pad_if_needed=True, padding_mode='constant'),
                transforms.Resize((224, 224))
            ])
        else:
            trans = transforms.Compose([
            transforms.CenterCrop((int(h - h * (1 - scale)**2), int(w - w * (1 - scale)**2))),
            transforms.RandomCrop((int(h * scale), int(w * scale)), pad_if_needed=True, padding_mode='constant'),
            # transforms.Resize((224, 224))
            ])

        img = trans(image)

        return img


class SegData(data.Dataset):
    def __init__(self, csv_file_path: str, img_dir: str, mask_dir: str, aug: A.Compose=None, input_transform=None, target_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        df = pd.read_csv(csv_file_path)
        self.img_id = list(df['image_id'])
        self.aug = aug
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_id[idx] + '.jpg')
        mask_path = os.path.join(self.mask_dir, self.img_id[idx] + '_segmentation.png')
#         image = Image.open(img_path)
#         mask = Image.open(mask_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = ImageOps.grayscale(mask)
        # TO DO: mask unsqueeze
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.aug:
            # another way: https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
            augmented = self.aug(image=image, mask=mask)
#             mask = self.transform(mask) # .unsqueeze(dim=0)
            image, mask = augmented['image'], augmented['mask']
        
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask