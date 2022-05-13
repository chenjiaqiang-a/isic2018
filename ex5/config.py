from noise.loss import *

CLASS_NUM = 7
tau, p, lamb, rho, freq = 0.5, 0.01, 5, 1.002, 1

class_weight = [1 for _ in range(CLASS_NUM)]


def get_config(exp_id: str):
    loss_id, noise_id = exp_id.split('-')
    # configure loss
    if loss_id == '1':
        criterion = FocalLoss(alpha=class_weight)
    elif loss_id == '2':
        criterion = SR(FocalLoss(alpha=class_weight), tau, p, lamb)
    elif loss_id == '3':
        criterion = GCELoss(num_classes=CLASS_NUM)
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

    return criterion, noise_rate

