import torch
import os
import gc


class Trainer(object):
    def __init__(self, device, log, model_name: str, optimizer=None, scheduler=None, grad_bound: float = 5., start_epoch: int = 0, best_score=0, checkpoint_model=None):
        """ trainer for segmentation tasks

        Args:
            device (torch.device)
            log (Logger): logfile
            model_name (str): name of the model
            optimizer (torch.nn.optim)
            scheduler (torch.nn.optim)
            grad_bound (float): max norm of the gradients
            start_epoch (int): initial epoch
            best_score (float): metric score for early stopping
            checkpoint_model (None or nn.Module): None - train from scratch; nn.Module - reload from checkpoint
        """
        self.device = device
        self.log = log
        self.model_name = model_name
        self.grad_bound = grad_bound
        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        # path to store checkpoint
        self.pth_check = os.path.join(
            'checkpoint', 'check_' + model_name + '.pth')

        if checkpoint_model == None:
            self.epoch = start_epoch
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.train_costs = []
            self.train_accs = []
            self.train_actual_accs = []
            self.val_costs = []
            self.val_accs = []
            self.val_actual_accs = []
            self.best_score = best_score
            self.patience = 0
        else:
            checkpoint = torch.load(self.pth_check)
            self.epoch = checkpoint['epoch'] + 1
            self.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.train_costs = checkpoint['train_costs']
            self.train_accs = checkpoint['train_accs']
            self.train_actual_accs = checkpoint['train_actual_accs']
            self.val_costs = checkpoint['val_costs']
            self.val_accs = checkpoint['val_accs']
            self.val_actual_accs = checkpoint['val_actual_accs']
            self.best_score = checkpoint['best_score']
            self.patience = checkpoint['patience']
            checkpoint_model.load_state_dict(checkpoint['model_state_dict'])

    def fit(self, model, train_loader, val_loader, criterion, rho: float, freq: int, max_epoch, test_period=5, early_threshold=10):
        size_train = len(train_loader)
        num_train = len(train_loader.dataset)
        size_val = len(val_loader)
        num_val = len(val_loader.dataset)
        model.train()

        for self.epoch in range(self.epoch, max_epoch):
            cost = 0
            acc = 0
            actual_acc = 0

            for x, clean_y, noisy_y in train_loader:
                x, clean_y, noisy_y = x.to(self.device), clean_y.to(
                    self.device), noisy_y.to(self.device)
                self.optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, noisy_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_bound) # 梯度裁剪
                self.optimizer.step()

                cost += loss.item()
                _, yhat = torch.max(z.data, 1)
                acc += (yhat == noisy_y).sum().item()
                actual_acc += (yhat == clean_y).sum().item()

            self.train_costs.append(cost/size_train)
            self.train_accs.append(acc/num_train)
            self.train_actual_accs.append(actual_acc/num_train)
            self.scheduler.step()

            gc.collect()

            if self.epoch % test_period == 0:
                model.eval()
                cost, acc, noisy_acc = 0, 0, 0
                with torch.no_grad():
                    for x, clean_y, noisy_y in val_loader:
                        x, clean_y, noisy_y = x.to(self.device), clean_y.to(
                            self.device), noisy_y.to(self.device)
                        z = model(x)
                        loss = criterion(z, clean_y)
                        cost += loss.item()
                        _, yhat = torch.max(z.data, 1)
                        acc += (yhat == clean_y).sum().item()
                        noisy_acc += (yhat == noisy_y).sum().item()

                self.val_costs.append(cost/size_val)
                self.val_accs.append(noisy_acc/num_val)
                self.val_actual_accs.append(acc/num_val)

                if self.val_actual_accs[-1] >= self.best_score:
                    self.best_score = self.val_actual_accs[-1]
                    self.patience = 0
                    save_state_dict(
                        model, name="{}_dict.pth".format(self.model_name))
                else:
                    self.patience += 1
                    if self.patience >= early_threshold:
                        break

                self.log.logger.info("Epoch:{:3d} train_cost: {:.4f}\ttrain_acc: {:.4f}\ta_acc: {:.4f}\tval_cost: {:.4f}\tval_acc: {:.4f}\tva_acc: {:.4f}".format(
                    self.epoch+1, self.train_costs[-1], self.train_accs[-1], self.train_actual_accs[-1], self.val_costs[-1], self.val_accs[-1], self.val_actual_accs[-1]))

                model.train()

            self.checkpoint(self.pth_check, model)

            # Adapt params of SR
            if freq != 0 and (self.epoch + 1) % freq == 0:
                criterion.lamb *= rho

        save_model(model, name=self.model_name+'.pkl')
        history = self.get_history()
        self.log.logger.info("Model has been saved at {}\n{}".format(
            self.model_name+'.pkl', history))
        return history

    def checkpoint(self, check_file, model):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'epoch': self.epoch,
            'train_costs': self.train_costs,
            'train_accs': self.train_accs,
            'train_actual_accs': self.train_actual_accs,
            'val_costs': self.val_costs,
            'val_accs': self.val_accs,
            'val_actual_accs': self.val_actual_accs,
            'best_score': self.best_score,
            'patience': self.patience
        }
        torch.save(checkpoint, check_file)

    def get_history(self):
        history = {
            'train_costs': self.train_costs,
            'train_accs': self.train_accs,
            'train_actual_accs': self.train_actual_accs,
            'val_costs': self.val_costs,
            'val_accs': self.val_accs,
            'val_actual_accs': self.val_actual_accs,
            'best_score': self.best_score
        }
        return history


def load_model(device, name='model.pkl'):
    """
    load model from ./model/name 加载网络
    """
    pth_model = os.path.join('model', name)
    assert os.path.exists(pth_model), "Model file doesn't exist!"
    model = torch.load(pth_model, map_location=device)
    print('Load {} on {} successfully.'.format(name, device))
    return model


def save_model(model, name='model.pkl'):
    """ 
    save model to ./model/name 保存网络
    """

    if not os.path.exists('model'):
      os.makedirs('model')

    pth_model = os.path.join('model', name)
    torch.save(model, pth_model)
    print('Model has been saved to {}'.format(pth_model))


def save_state_dict(model, name='state_dict.pth'):
    """ 
    save state dict to ./model/temp/name 保存网络参数
    """

    model_dir = os.path.join('model', 'temp')
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    pth_dict = os.path.join(model_dir, name)
    torch.save(model.state_dict(), pth_dict)
    print('state dict has been saved to {}'.format(pth_dict))


def load_state_dict(model, device, name='state_dict.pth'):
    """ 
    load model parmas from state_dict 加载网络参数
    """
    pth_dict = os.path.join('model', 'temp', name)
    assert os.path.exists(pth_dict), "State dict file doesn't exist!"
    model.load_state_dict(torch.load(pth_dict, map_location=device))
    return model
