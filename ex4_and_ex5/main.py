import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import OrderedDict
import random
from config import *
from noise.logger import Logger
from noise.train import Trainer, load_state_dict
from noise.evaluation import evaluation


# Argparse
parser = argparse.ArgumentParser(description='Robust Learning via Sparse Regularization')
parser.add_argument('--exp_id', type=str, default='1-1', help='Experiment ID')
parser.add_argument('--model_id', type=int, default=1, help='Model ID of the experiment')
parser.add_argument('--message', type=str, default='')
# learning settings
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='the number of worker for loading data')
parser.add_argument('--grad_bound', type=float, default=5., help='the gradient norm bound')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('-c', '--checked', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()


# Setting the environment
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available()  else 'cpu'

random.seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)


# log file
model_id = args.model_id
exp_id = args.exp_id
model_name = exp_id + '-' + str(model_id)
log = Logger(mode='exp', title=exp_id)
log.logger.info("{} | {} | Batch Size: {}, Seed: {}".format(model_name, args.message, args.batch_size, args.seed))

# Experiment Setting
criterion, noise_type, noise_rate, rho, freq = get_config(exp_id)

# Dataset
train_loader = generate_data('train', noise_type, noise_rate, args.batch_size, args.num_workers, args.seed)
valid_loader = generate_data('valid', noise_type, noise_rate, args.batch_size*2, args.num_workers, args.seed)


# Model
model = models.densenet201(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
    ('fc0', nn.Linear(1920, 256)),
    ('norm0', nn.BatchNorm1d(256)),
    ('relu0', nn.ReLU(inplace=True)),
    ('fc1', nn.Linear(256, 7))
]))
model.classifier = classifier
model.to(device)
    

# Train
init_lr = 1e-3
weight_decay = 1e-4
max_epoch = 100
test_period = 1
early_threshold = 40

optimizer = optim.AdamW(model.classifier.parameters(), lr=init_lr, betas=(0.9, 0.999), weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)
# scheduler = StepLR(optimizer, gamma=0.1, step_size=25)

if args.checked == 1:
    trainer = Trainer(device, log, model_name, optimizer, scheduler, checkpoint_model=model)
else:
    trainer = Trainer(device, log, model_name, optimizer, scheduler, checkpoint_model=None)


history = trainer.fit(model, train_loader, valid_loader, criterion, rho, freq, 10, test_period, early_threshold)
# unfreeze layers
for param in model.parameters():
    param.requires_grad = True
history = trainer.fit(model, train_loader, valid_loader, criterion, rho, freq, max_epoch, test_period, early_threshold)


# Evaluation
test_loader = generate_data('test', noise_type, noise_rate, args.batch_size*2, args.num_workers, args.seed)
log.logger.info(evaluation(model, test_loader))

log.logger.info('Best Model')
model = load_state_dict(model, device, name='{}_dict.pth'.format(trainer.model_name))
log.logger.info(evaluation(model, test_loader))