import warnings
warnings.filterwarnings('ignore')

from data_provider.data_factory import data_provider
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.optim import lr_scheduler 
from torch.utils.data.distributed import DistributedSampler

import os
import time
import sys


import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext
from math import sqrt


#重写print，同时输出到终端、并保存到文件中
def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']
    return wrapper
print = custom_print_decorator(print)




class Exp_Main(object):
    def __init__(self, args, setting):
        self.args = args
        self.ddp = self.args.use_multi_gpu and (not args.debug)
        
        # 设置设备和分布式环境
        if self.ddp:
            self.device_id = dist.get_rank() % torch.cuda.device_count()#当前进程在哪个GPU上
            self.is_main_process = dist.get_rank() == 0
        else:
            self.device_id = 0
            self.is_main_process = True
        
        # 创建日志权重保存点，只在主进程中创建目录
        self.path = os.path.join(self.args.checkpoints, setting)
        if self.is_main_process:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        
        
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).to(self.device_id)

        #DDP配置
        if self.ddp:
            model = DDP(model, device_ids=[self.device_id], find_unused_parameters=True)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.ddp)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 只在分布式训练时进行同步
        if self.ddp:
            # torch.cuda.set_device(self.device_id)
            dist.barrier()
        self.model = self._build_model()
        
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        #打印参数信息
        if self.is_main_process:
            pytorch_total_params = sum(p.numel() for p in self.model.parameters())
            print("Parameters number {} M".format(
                pytorch_total_params/1e6), folder=self.path)
            print("{} steps for each epoch".format(train_steps), folder=self.path)

        #创建梯度缩放器，防止FP16梯度下溢出现的梯度消失问题，通过放大损失值使得小梯度在FP16下也能有有效表示
        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        
        #学习率调度器，OneCycle调度器：学习率先增加后减少，pct_start为学习率增加时段的百分比
        #DDP等价于更大的batch=batch_size*acc_steps, 因此由经验公式学习率：
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps//self.args.acc_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate*sqrt(self.args.acc_steps))

        for epoch in range(self.args.train_epochs):
            train_loss = self.train_one_epoch(train_loader, model_optim, criterion, epoch, scaler, scheduler)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


            best_model_path = self.path + '/' + 'checkpoint.pth'
            # 如果使用分布式训练，确保所有进程同步
            if dist.is_initialized():
                dist.barrier()
                # 只在主进程中加载最佳模型
                if self.is_main_process:
                    torch.save(self.model.module.state_dict(), best_model_path)
                dist.barrier()
                # 所有进程加载同一个模型
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device_id}
                self.model.module.load_state_dict(torch.load(best_model_path, map_location=map_location))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

            
    def train_one_epoch(self, train_loader, model_optim, criterion, epoch, scaler=None, scheduler=None):
        iter_count = 0
        train_loss = []
        acc_steps = self.args.acc_steps  # 使用args中的acc_steps

        # 设置dataloader的epoch
        if self.ddp and hasattr(train_loader, 'sampler'):
            dist.barrier()
            train_loader.sampler.set_epoch(epoch)

        self.model.train()
        epoch_time = time.time()
        iter_time = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # 只在累积开始时清零梯度
            if i % acc_steps == 0:
                model_optim.zero_grad(set_to_none=True)  # 使用优化器的zero_grad
            
            iter_count += 1
            
            # 数据准备
            batch_x = batch_x.float().to(self.device_id)
            batch_y = batch_y.float().to(self.device_id)
            batch_x_mark = batch_x_mark.float().to(self.device_id)
            batch_y_mark = batch_y_mark.float().to(self.device_id)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device_id)

            # 确定是否跳过梯度同步
            is_last_in_acc = (i % acc_steps == acc_steps - 1) or (i + 1) == len(train_loader)
            
            #如果不是最后一个acc_it，则不进行梯度同步
            context = self.model.no_sync if self.ddp and not is_last_in_acc else nullcontext
            with context():
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        # 计算损失
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device_id)
                        #TODO 整明白损失函数计算
                        loss = criterion(outputs, batch_y)
                        loss = loss / acc_steps
                        train_loss.append(loss.item() * acc_steps)
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                else:
                    # 前向传播
                    outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # 计算损失
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device_id)
                    loss = criterion(outputs, batch_y)
                    loss = loss / acc_steps
                    train_loss.append(loss.item() * acc_steps)
                    
                    # 反向传播
                    loss.backward()

            # 打印日志
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f} | using: {3}".format(
                    i + 1, epoch + 1, loss.item(), time.time() - iter_time))
                iter_count = 0
                iter_time = time.time()
            
            # 累积结束后更新参数
            if is_last_in_acc:
                if self.args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    model_optim.step()
                
                # 每个batch结束都更新学习率
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
                if self.ddp:
                    dist.barrier()
            

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        return np.average(train_loss)
        
    def _forward_pass(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """封装模型前向传播逻辑"""
        if 'Linear' in self.args.model or 'TST' in self.args.model:
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device_id)
                batch_y_mark = batch_y_mark.float().to(self.device_id)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device_id)
                
                # 使用封装的前向传播方法
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device_id)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            if self.is_main_process:
                print('loading model')
            if self.args.use_multi_gpu and self.args.use_gpu and dist.is_initialized():
                # 分布式环境中加载模型
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device_id}
                self.model.module.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=map_location))
            else:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.float().to(self.device_id)

                batch_x_mark = batch_x_mark.float().to(self.device_id)
                batch_y_mark = batch_y_mark.float().to(self.device_id)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device_id)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device_id)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        if self.is_main_process:
            print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n')
            f.write('\n')
            f.close()

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
            np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)
            # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            if self.args.use_multi_gpu and self.args.use_gpu and dist.is_initialized():
                # 分布式环境中加载模型
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device_id}
                self.model.module.load_state_dict(torch.load(best_model_path, map_location=map_location))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device_id)
                batch_y_mark = batch_y_mark.float().to(self.device_id)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device_id)
                
                # 使用封装的前向传播方法
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.is_main_process:
            np.save(folder_path + 'real_prediction.npy', preds)

        return
