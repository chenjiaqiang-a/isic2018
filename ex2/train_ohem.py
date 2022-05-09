import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from net import arlnet50
from utils import Logger, ISIC2018Dataset, save_model

# 参数设置
RUN_FOLDER = './run/ohem'
RUN_ID = '0001'
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, "images"))
    os.makedirs(os.path.join(RUN_FOLDER, "models"))
LOGGER = Logger(RUN_FOLDER, RUN_ID)
LOGGER.info(f"Ex2: ohem {RUN_ID} run by Chen")

# 数据预处理
BATCH_SIZE = 64
NUM_WORKERS = 4
train_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.4, 1), ratio=(3 / 4, 4 / 3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ISIC2018Dataset(
    csv_file_path='./data/ISIC2018/Train_GroundTruth.csv',
    img_dir='./data/ISIC2018/ISIC2018_Task3_Training_Input',
    transform=train_trans
)

valid_dataset = ISIC2018Dataset(
    csv_file_path='./data/ISIC2018/ISIC2018_Task3_Validation_GroundTruth.csv',
    img_dir='./data/ISIC2018/ISIC2018_Task3_Validation_Input',
    transform=valid_trans
)

train_iter = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        drop_last=True,
                        num_workers=NUM_WORKERS)
valid_iter = DataLoader(valid_dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS)

LOGGER.info(f"Train on isic2018 dataset with {len(train_dataset)} train samples, "
            f"{len(valid_dataset)} valid samples")

# 模型准备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = arlnet50(num_classes=train_dataset.num_classes, pretrained=True)
model = model.to(DEVICE)

LOGGER.info("Using arlnet50 model with pretrained weights")

# 训练准备
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
EPOCHS = 100
VALID_PERIOD = 1
EARLY_THRESHOLD = 40
HARD_EXAMPLE_RATE = 1/2

loss_fn = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

LOGGER.info(f"Using loss function: {loss_fn}")

# 开始训练模型
LOGGER.info("Start training...")

train_l = []
train_acc = []
valid_l = []
valid_acc = []
max_acc = 0
epoch_counter = EARLY_THRESHOLD
num_hard_examples = int(BATCH_SIZE * HARD_EXAMPLE_RATE)
for epoch in range(EPOCHS):
    num_data = 0
    correct = 0
    loss_sum = 0
    for X, y in train_iter:
        num_data += X.shape[0]
        X, y = X.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(X)
            loss = loss_fn(out, y)
            _, indices = torch.sort(loss, descending=True)
            hard_idx = indices[:num_hard_examples]

            loss_sum += loss.sum().cpu().detach().item()
            yhat = out.argmax(dim=1)
            correct += (yhat == y).sum().cpu().detach().item()

        model.train()
        hard_X, hard_y = X[hard_idx], y[hard_idx]
        out = model(hard_X)
        loss = loss_fn(out, hard_y)

        loss.mean().backward()
        optimizer.step()

    loss = loss_sum / num_data
    train_l.append(loss)
    acc = correct / num_data
    train_acc.append(acc)

    scheduler.step()

    if (epoch + 1) % VALID_PERIOD == 0:
        num_data = 0
        correct = 0
        loss_sum = 0
        model.eval()
        for X, y in valid_iter:
            optimizer.zero_grad()
            num_data += X.shape[0]
            X, y = X.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                out = model(X)
                loss = loss_fn(out, y)

                loss_sum += loss.sum().cpu().detach().item()
                yhat = out.argmax(dim=1)
                correct += (yhat == y).sum().cpu().detach().item()

        loss = loss_sum / num_data
        valid_l.append(loss)
        acc = correct / num_data
        valid_acc.append(acc)

        LOGGER.info("Epoch {:03d} --- train loss: {:.4f} train acc: {:.4f}\ttest loss: {:.4f} test acc: {:.4f}".format(
            epoch + 1, train_l[-1], train_acc[-1], valid_l[-1], valid_acc[-1]
        ))

        if acc > max_acc:
            max_acc = acc
            save_model(model, os.path.join(RUN_FOLDER, "models"), RUN_ID + "model.pkl")
            epoch_counter = EARLY_THRESHOLD
        else:
            epoch_counter -= 1

        if epoch_counter == 0:
            LOGGER.info("Early Stopped!")
            break

pickle.dump({
    "train_loss": train_l,
    "train_acc": train_acc,
    "valid_loss": valid_l,
    "valid_acc": valid_acc
}, open(os.path.join(RUN_FOLDER, RUN_ID+"obj.pkl"), "wb"))
