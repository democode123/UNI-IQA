import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
import scipy.stats
from ImageDataset import ImageDataset
from model import UNI_IQA
from MNL_Loss import Fidelity_Loss
from Transformers import AdaptiveResize
import numpy as np
class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.config = config

        self.train_transform = transforms.Compose([
            # transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.train_batch_size = config.batch_size
        self.test_batch_size = 1

        self.train_data = ImageDataset(
            csv_file=os.path.join(config.trainset,'splits2', '1', config.train_txt),
            img_dir=config.trainset,
            transform=self.train_transform,
            test=False)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=0)

        # testing set configuration
        self.live_data = ImageDataset(
            csv_file=os.path.join(config.live_set, '1', 'live_test.txt'),
            img_dir=config.live_set,
            transform=self.test_transform,
            test=True)

        self.live_loader = DataLoader(self.live_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=0)

        self.QACS_data = ImageDataset(
            csv_file=os.path.join(config.QACS_set, '1', 'QACS_test.txt'),
            img_dir=config.QACS_set,
            transform=self.test_transform,
            test=True)

        self.QACS_loader = DataLoader(self.QACS_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")

        # initialize the model
        self.model = UNI_IQA(config)
        self.model = nn.DataParallel(self.model).cuda()

        self.model.to(self.device)
        self.model_name = type(self.model).__name__
        # print(self.model)

        # loss function

        self.loss_fn = Fidelity_Loss()

        self.loss_fn.to(self.device)

        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=5e-4)

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results_srcc = {'live': []}
        self.test_results_plcc = {'live': []}
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        # if config.resume or not config.train:
        #     if config.ckpt:
        #         ckpt = os.path.join(config.ckpt_path, config.ckpt)
        #     else:
        #         ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
        #     self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch - 1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)
    def fit(self):
            for epoch in range(self.start_epoch, self.max_epochs):
                    _ = self._train_single_epoch(epoch)
                    self.scheduler.step()

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        # running_std_loss = 0 if epoch == 0 else self.train_std_loss[-1]
        loss_corrected = 0.0
        # std_loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        # self.scheduler.step()
        for step, sample_batched in enumerate(self.train_loader, 0):

            if step < self.start_step:
                continue

            sci1, sci2, yb_sci,ni1, ni2, yb_ni = sample_batched['SCI1'], sample_batched['SCI2'], sample_batched['yb_SCI'],sample_batched['NI1'], sample_batched['NI2'], sample_batched['yb_NI']
            sci1 = Variable(sci1)
            sci2 = Variable(sci2)
            # yb_sci = torch.tensor([item.cpu().detach().numpy() for item in yb_sci])
            yb_sci = Variable(yb_sci).view(-1,1)
            sci1 = sci1.to(self.device)
            sci2 = sci2.to(self.device)
            yb_sci = yb_sci.to(self.device)

            ni1 = Variable(ni1)
            ni2 = Variable(ni2)
            # yb_ni = torch.tensor([item.cpu().detach().numpy() for item in yb_ni])
            yb_ni = Variable(yb_ni).view(-1,1)
            ni1 = ni1.to(self.device)
            ni2 = ni2.to(self.device)
            yb_ni = yb_ni.to(self.device)

            self.optimizer.zero_grad()

            x1, y1 = self.model(sci1, ni1)
            x2, y2 = self.model(sci2, ni2)
            x_diff = x1 - x2
            y_diff = y1 - y2

            p = y_diff
            constant = torch.sqrt(torch.Tensor([2])).to(self.device)
            p = 0.5 * (1 + torch.erf(p / constant))
            loss1 = self.loss_fn(p, yb_sci.detach())

            q = x_diff
            constant = torch.sqrt(torch.Tensor([2])).to(self.device)
            q = 0.5 * (1 + torch.erf(q / constant))
            loss2 = self.loss_fn(q, yb_ni.detach())

            self.loss = loss1 + loss2

            self.loss.backward()
            self.optimizer.step()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d / %d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()

        self.train_loss.append(loss_corrected)

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results_srcc': self.test_results_srcc,
                'test_results_plcc': self.test_results_plcc,
            }, model_name)
        # if (epoch+1) % self.epochs_per_eval == 0:
        if (epoch + 1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            test_results_srcc, test_results_plcc = self.eval()
            self.test_results_srcc['live'].append(test_results_srcc['live'])
            self.test_results_plcc['live'].append(test_results_plcc['live'])

            out_str = 'Testing: LIVE SRCC: {:.4f}'.format(
                test_results_srcc['live'])
            out_str2 = 'Testing: LIVE PLCC: {:.4f}'.format(
                test_results_plcc['live'])

            print(out_str)
            print(out_str2)

        return self.loss.data.item()



    def eval(self):
        srcc = {}
        plcc = {}

        self.model.eval()
        if self.config.eval_live:
            q_mos = []
            q_hat = []
            for step, sample_batched in enumerate(self.live_loader, 0):
                x, y = sample_batched['I'], sample_batched['mos']
                x = Variable(x)
                x = x.to(self.device)

                # if self.config.std_modeling:
                #     y_bar, _ = self.model(x)
                # else:
                _,y_bar = self.model(x,x)
                y_bar.cpu()
                # y = torch.tensor(float(y))
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            srcc['live'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['live'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
        else:
            srcc['live'] = 0
            plcc['live'] = 0

        if self.config.eval_QACS:
            q_mos = []
            q_hat = []
            for step, sample_batched in enumerate(self.QACS_loader, 0):
                x, y = sample_batched['I'], sample_batched['mos']
                x = Variable(x)
                x = x.to(self.device)

                _, y_bar = self.model(x, x)
                y_bar.cpu()
                q_mos.append(y.data.numpy())
                q_hat.append(y_bar.cpu().data.numpy())

            srcc['QACS'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
            plcc['QACS'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
        else:
            srcc['QACS'] = 0
            plcc['QACS'] = 0

        return srcc, plcc
    @staticmethod
    def _save_checkpoint(state, filename):
        torch.save(state, filename)