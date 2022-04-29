import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from net import resnet18
from utils import ISIC2018Dataset, save_model

BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.2

train_trans = transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_trans = transforms.Compose([
    transforms.CenterCrop((450, 450)),
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ISIC2018Dataset(
    csv_file_path='./data/ISIC2018/Train_GroundTruth.csv',
    img_dir='./data/ISIC2018/ISIC2018_Task3_Training_Input',
    transform=train_trans
)

test_dataset = ISIC2018Dataset(
    csv_file_path='./data/ISIC2018/Test_GroundTruth.csv',
    img_dir='./data/ISIC2018/ISIC2018_Task3_Training_Input',
    transform=test_trans
)

train_iter = DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        drop_last=True,
                        num_workers=NUM_WORKERS)
test_iter = DataLoader(test_dataset,
                       batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS)

net = resnet18()
net = net.to(DEVICE)
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optim = torch.optim.Adam(net.parameters(),
                         lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY)

train_l = []
train_acc = []
test_l = []
test_acc = []
for epoch in range(EPOCHS):
    correct = 0
    num_data = 0
    losses = 0
    net.train()
    for X, y in train_iter:
        num_data += X.shape[0]
        X, y = X.to(DEVICE), y.to(DEVICE)

        out = net(X)
        l = loss_fn(out, y)

        l.backward()
        optim.step()

        losses += l.cpu().detach().item()
        yhat = out.argmax(dim=1)
        correct += (yhat == y).sum().cpu().detach().item()

    loss = losses / num_data
    acc = correct / num_data
    train_l.append(loss)
    train_acc.append(acc)

    correct = 0
    num_data = 0
    losses = 0
    net.eval()
    for X, y in test_iter:
        num_data += X.shape[0]
        X, y = X.to(DEVICE), y.to(DEVICE)

        out = net(X)
        l = loss_fn(out, y)

        losses += l.cpu().detach().item()
        yhat = out.argmax(dim=1)
        correct += (yhat == y).sum().cpu().detach().item()

    loss = losses / num_data
    acc = correct / num_data
    test_l.append(loss)
    test_acc.append(acc)

    print("Epoch {:03d} --- train loss: {:.4f} train acc: {:.4f}\ttest loss: {:.4f} test acc: {:.4f}".format(
        epoch+1, train_l[-1], train_acc[-1], test_l[-1], test_acc[-1]
    ))

save_model(model=net)
