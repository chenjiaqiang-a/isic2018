from noise.loss import *
from torchvision import transforms
from noise.data import NoisyISIC2018
from torch.utils import data

__all__ = ["get_config", "generate_data"]


CLASS_NUM = 7
tau, p, lamb, rho, freq = 0.5, 0.01, 5, 1.002, 1
# TAU, P, LAMB, RHO, FREQ = 0.5, 0.01, 5, 1.002, 1


class_weight = [1 for _ in range(CLASS_NUM)]


def get_config(exp_id: str):
    loss_id, noise_id = exp_id.split('-')
    
    # configure loss
    if loss_id == '1':
        criterion = FocalLoss(alpha=class_weight)
        freq = 0
    elif loss_id == '2':
        criterion = SR(FocalLoss(alpha=class_weight), tau, p, lamb)
    elif loss_id == '3':
        criterion = GCELoss(num_classes=CLASS_NUM)
        freq = 0
    else:
        raise ValueError("Experiment ID doesn't exist")

    # configure noisy labels
    if noise_id == '1':
        noise_rate = 0
    elif noise_id == '2':
        noise_rate = .1
    elif noise_id == '3':
        noise_rate = .4
    else:
        raise ValueError("Experiment ID doesn't exist")

    return criterion, noise_rate, rho, freq


# transforms
trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.4, 1), ratio=(3/4, 4/3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trans_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def generate_data(mode, noise_rate, batch_size, num_workers, random_seed):
    if mode == 'train':
        train_data = NoisyISIC2018(ann_file='../../Robust-Skin-Lesion-Diagnosis/Data/2018/Train_GroundTruth.csv',
                                img_dir='../../Robust-Skin-Lesion-Diagnosis/Data/2018/ISIC2018_Task3_Training_Input',
                                transform=trans_train, noise_type='asymmetric', noise_rate=noise_rate, random_state=random_seed)
        data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    elif mode == 'test':
        test_data = NoisyISIC2018(ann_file='../../Robust-Skin-Lesion-Diagnosis/Data/2018/Test_GroundTruth.csv',
                                img_dir='../../Robust-Skin-Lesion-Diagnosis/Data/2018/ISIC2018_Task3_Training_Input',
                                transform=trans_test, noise_type='symmetric', noise_rate=noise_rate, random_state=random_seed)
        data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elif mode == 'valid':
        valid_data = NoisyISIC2018(ann_file='../../Robust-Skin-Lesion-Diagnosis/Data/2018/ISIC2018_Task3_Validation_GroundTruth.csv',
                                img_dir='../../Robust-Skin-Lesion-Diagnosis/Data/2018/ISIC2018_Task3_Validation_Input',
                                transform=trans_test, noise_type='symmetric', noise_rate=noise_rate, random_state=random_seed)
        data_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return data_loader