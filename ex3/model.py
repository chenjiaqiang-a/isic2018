from pickletools import optimize
import torch
import os, tqdm, gc
from .evaluation import pixel_accuracy, mIoU


# train
class SegTrain(object):
    def __init__(self, device, log, model_name: str, optimizer=None, scheduler=None, start_epoch: int = 0, best_score=0, checkpoint_model=None):
        """ trainer for segmentation tasks

        Args:
            device (torch.device)
            log (Logger): logfile
            model_name (str): name of the model
            optimizer (torch.nn.optim)
            scheduler (torch.nn.optim)
            start_epoch (int): initial epoch.
            best_score (float): metric score for early stopping
            checkpoint_model (None or nn.Module): None - train from scratch; nn.Module - reload from checkpoint
        """
        self.device = device
        self.log = log
        self.model_name = model_name
        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        # path to store checkpoint
        self.pth_check = os.path.join('checkpoint', 'check_' + model_name + '.pth')

        if checkpoint_model == None:
            self.epoch = start_epoch
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.costs = []
            self.train_accs = []
            self.train_ious = []
            self.val_accs = []
            self.val_ious = []
            self.best_score = best_score
            self.patience = 0
        else:
            checkpoint = torch.load(self.pth_check)
            self.epoch = checkpoint['epoch'] + 1
            self.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.costs = checkpoint['costs']
            self.train_accs = checkpoint['train_accs']
            self.train_ious = checkpoint['train_ious']
            self.val_accs = checkpoint['val_accs']
            self.val_ious = checkpoint['val_ious']
            self.best_score = checkpoint['best_score']
            self.patience = checkpoint['patience']
            checkpoint_model.load_state_dict(checkpoint['model_state_dict'])

    def fit(self, model, train_loader, val_loader, criterion, max_epoch, test_period=5, early_threshold=10):
        size_train = len(train_loader)
        size_val = len(val_loader)
        model.train()

        for self.epoch in range(self.epoch, max_epoch):
            cost = 0
            iou_score = 0
            pixel_acc = 0

            for image, mask in train_loader:
                image, mask = image.to(self.device), mask.to(self.device)
                self.optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, mask)
                loss.backward()
                self.optimizer.step()

                iou_score += mIoU(output, mask)
                pixel_acc += pixel_accuracy(output, mask)
                cost += loss.item()

            cost /= size_train
            self.costs.append(cost)
            self.train_accs.append(pixel_acc/size_train)
            self.train_ious.append(iou_score/size_train)
            self.scheduler.step()
            
            # del image, mask, loss
            gc.collect()

            if self.epoch % test_period == 0:
                # self.eval(model, val_loader)
                model.eval()
                iou_score = 0
                pixel_acc = 0
                with torch.no_grad():
                    for image, mask in val_loader:
                        image, mask = image.to(self.device), mask.to(self.device)
                        output = model(image)
                        iou_score += mIoU(output, mask)
                        pixel_acc += pixel_accuracy(output, mask)

                self.val_accs.append(pixel_acc/size_val)
                self.val_ious.append(iou_score/size_val)
                
                if self.val_ious[-1] >= self.best_score:
                    self.best_score = self.val_ious[-1]
                    self.patience = 0
                    save_state_dict(model, name="{}_dict.pth".format(self.model_name))
                else:
                    self.patience += 1
                    if self.patience >= early_threshold:
                        break
                
                self.log.logger.info("Epoch:{:3d} cost: {:.4f}\ttrain_acc: {:.4f}\ttrain_iou: {:.4f}\tval_acc: {:.4f}\tval_iou: {:.4f}".format(self.epoch+1, cost, self.train_accs[-1], self.train_ious[-1], self.val_accs[-1], self.val_ious[-1]))
                
                model.train()

            self.checkpoint(self.pth_check, model)

        save_model(model, name=self.model_name+'.pkl')
        history = self.get_history()
        self.log.logger.info("Model has been saved at {}\n{}".format(self.model_name+'.pkl', history))
        return history

    @torch.no_grad()
    def eval(self, model, val_loader):
        size_val = len(val_loader)
        model.eval()
        iou_score = 0
        pixel_acc = 0
        for image, mask in val_loader:
            image, mask = image.to(self.device), mask.to(self.device)
            output = model(image)
            iou_score += mIoU(output, mask)
            pixel_acc += pixel_accuracy(output, mask)

        pixel_acc /= size_val
        iou_score /= size_val
        return pixel_acc, iou_score


    def checkpoint(self, check_file, model):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'epoch': self.epoch,
            'costs': self.costs,
            'train_accs': self.train_accs,
            'train_ious': self.train_ious,
            'val_accs': self.val_accs,
            'val_ious': self.val_ious,
            'best_score': self.best_score,
            'patience': self.patience
        }
        torch.save(checkpoint, check_file)
        

    def get_history(self):
        history = {
            'costs': self.costs,
            'train_accs': self.train_accs,
            'train_ious': self.train_ious,
            'val_accs': self.val_accs,
            'val_ious': self.val_ious
        }
        return history
    
    def print_history(self): 
        return "costs = {}\ntrain_accs = {}\ntrain_ious = {}\nval_accs = {}\nval_ious = {}".format(self.costs, self.train_accs, self.train_ious, self.val_accs, self.val_ious)




# store model

def load_model(device, path='.', name='model.pkl'):
    """
    load model from path/model/name 加载网络
    """
    pth_model = os.path.join(path, 'model', name)
    assert os.path.exists(pth_model), "Model file doesn't exist!"
    model = torch.load(pth_model, map_location=device)
    print('Load {} on {} successfully.'.format(name, device))
    return model
    

def save_model(model, path='.', name='model.pkl'):
    """ 
    save model to path/model/name 保存网络
    """

    model_dir = os.path.join(path, 'model')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      
    pth_model = os.path.join(model_dir, name)
    torch.save(model, pth_model)
    print('Model has been saved to {}'.format(pth_model))


def save_state_dict(model, path='.', name='state_dict.pth'):
    """ 
    save state dict to path/model/temp/name 保存网络参数
    """

    model_dir = os.path.join(path, 'model', 'temp')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      
    pth_dict = os.path.join(model_dir, name)
    torch.save(model.state_dict(), pth_dict)
    print('state dict has been saved to {}'.format(pth_dict))
    
    
def load_state_dict(model, device, path='.', name='state_dict.pth'):
    """ 
    load model parmas from state_dict 加载网络参数
    """
    pth_dict = os.path.join(path, 'model', 'temp', name)
    assert os.path.exists(pth_dict), "State dict file doesn't exist!"
    model.load_state_dict(torch.load(pth_dict, map_location=device))
    return model


# checkpoint

def check_train(log, model, optimizer, epoch, scheduler=None, pth_check='ch_training.pth', verbose=False):
    """ save training checkpoint
        保存训练参数：model, epoch, optimizer, scheduler

    Args:
        log (Logger)
        pth_check (str): path to store the checkpoint.
    """
    check_dir = 'checkpoint'
    if not os.path.exists(check_dir):
      os.makedirs(check_dir)
    pth_check = os.path.join(check_dir, pth_check)

    if verbose:
        log.logger.info("Saving training checkpoint at {}".format(pth_check))
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, pth_check)


def check_eval(log, costs, train_accs, test_accs, b_accs, f1_scores, auces, pth_check='ch_eval.pth', verbose=True):
    """ saving evaluation checkpoint
        保存训练过程的cost, accs, f1-score, auc

    Args:
        log (Logger)
        pth_eval (str): path to store the checkpoint.
        verbose: whether showing details
    """
    check_dir = 'checkpoint'
    if not os.path.exists(check_dir):
      os.makedirs(check_dir)
    pth_check = os.path.join(check_dir, pth_check)
    
    if verbose:
        log.logger.info("Saving training checkpoint at {}".format(pth_check))
    checkpoint = {
        'costs': costs,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'b_accs': b_accs,
        'f1_scores': f1_scores,
        'auces': auces
    }

    if verbose:
        for key in checkpoint.keys(): 
            log.logger.info('{} = {}\n'.format(key, checkpoint[key]))

    torch.save(checkpoint, pth_check)


def load_train(log, model, optimizer, scheduler=None, pth_check=None):
    """ initialize or load training process from checkpoint
        从checkpoint加载训练状态，pth_check为None时，进行初始化

    Args:
        log (Logger)
        pth_check (str): path of training checkpoint file. e.g. 'ch_training.pth'. (Default: None - 初始化)

    Returns:
        start epoch
    """
    if pth_check == None:
        return 0
    
    pth_check = os.path.join('checkpoint', pth_check)
    log.logger.info("Reloading training checkpoint from {}".format(pth_check))
    checkpoint = torch.load(pth_check)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return start_epoch

def load_eval(log, pth_check=None):
    """ initialize or load evaluation from checkpoint
        从checkpoint加载之前训练过程的模型表现，pth_check为None时，进行初始化

    Args:
        log (Logger)
        pth_check (str): path of eval checkpoint file. e.g. 'ch_eval.pth'

    Returns:
        costs, train_accs, test_accs, b_accs, f1_scores, auces
    """

    if pth_check == None:
        return [], [], [], [], [], []

    pth_check = os.path.join('checkpoint', pth_check)
    log.logger.info("Reloading evaluation checkpoint from {}".format(pth_check))
    checkpoint = torch.load(pth_check)

    costs = checkpoint['costs']
    train_accs = checkpoint['train_accs']
    test_accs = checkpoint['test_accs']
    b_accs = checkpoint['b_accs']
    f1_scores = checkpoint['f1_scores']
    auces = checkpoint['auces']
    
    return costs, train_accs, test_accs, b_accs, f1_scores, auces
