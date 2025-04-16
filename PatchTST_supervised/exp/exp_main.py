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



class Exp_Main(object):
    def __init__(self, args, setting):
        self.args = args
        self.ddp = args.ddp
        
        # 设置设备和分布式环境
        if self.ddp:
            self.device = dist.get_rank() % torch.cuda.device_count()#当前进程在哪个GPU上
            self.is_main_process = dist.get_rank() == 0
        else:
            # self.device = 0 # 不要硬编码为 0
            # 使用 run_longExp.py 中为 test 实例设置的 device
            self.device = self.args.device 
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
        self.model = model_dict[self.args.model].Model(self.args).to(self.device) # 在用DDP设置通信之前，必须先把model移动到对应GPU设备上

        return self.model

    def _get_data(self, flag):
        #DDP配置，在data_provider中设置采样器sampler
        data_set, data_loader = data_provider(self.args, flag, self.ddp)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _forward_pass(self, batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if 'Linear' in self.args.model or 'TST' in self.args.model:
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        target_y = batch_y[:, -self.args.pred_len:, f_dim:]
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        return outputs, target_y
    
    def train(self):
        #1.读取数据
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #2. 加载模型(DDP模式下需要在优化器创建之后再用DDP包装)
        self.model = self._build_model()
        
        #3. 设置优化器、学习率调度器、损失函数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_steps = len(train_loader)
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        
        # DDP配置,DDP包装模型必须在优化器之后
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)

        #4. 设置日志输出参数，打印开始训练标识
        epoch_time = time.time()        #计算每个epoch的时间
        if self.is_main_process:
            pytorch_total_params = sum(p.numel() for p in self.model.parameters())
            print("Parameters number {} M".format(
                pytorch_total_params/1e6))
            print("{} steps for each epoch".format(train_steps))

        #5. 开始训练
        for epoch in range(self.args.train_epochs):
            #5.1 iter迭代计数器设置、ddp的损失计算设置
            iter_count = 0
            epoch_total_loss_sum = torch.tensor(0.0).to(self.device)
            epoch_total_samples = 0
            self.model.train()
            
            #5.2 ddp同步设备
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch)
                dist.barrier()

            #5.3 开始迭代
            time_start = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                #5.3.1 每个iter开始时梯度清零（累计梯度时候加判断条件）
                iter_count += 1
                model_optim.zero_grad()

                #5.3.2前向传播
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, target_y = self._forward_pass(batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark)

                #5.3.3 在iter内部计算梯度，反传梯度，更新模型参数
                loss = criterion(outputs, target_y)
                current_batch_size = target_y.size(0)
                epoch_total_loss_sum += loss.item() * current_batch_size
                epoch_total_samples += current_batch_size
                loss.backward()
                model_optim.step()

                #5.3.4在每个优化器 step 后调用 scheduler.step()
                scheduler.step()

                #在iter循环过程中打印输出信息
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} |use time:{3:.4f}".format(i + 1, epoch + 1, loss.item(), time.time()-time_start))
                    iter_count = 0
                    time_start = time.time() # 更新 time_start
                
            # 6. ddp模式下计算所有进程上的平均损失用于输出
            if self.args.ddp:
                # Synchronize epoch loss and samples across all ranks
                dist.all_reduce(epoch_total_loss_sum, op=dist.ReduceOp.SUM)
                epoch_total_samples_tensor = torch.tensor(epoch_total_samples).to(self.device)
                dist.all_reduce(epoch_total_samples_tensor, op=dist.ReduceOp.SUM)
                epoch_total_samples = epoch_total_samples_tensor.item() # Get the global total samples
            avg_epoch_train_loss = epoch_total_loss_sum / epoch_total_samples if epoch_total_samples > 0 else 0
            avg_epoch_train_loss = avg_epoch_train_loss.item() if isinstance(avg_epoch_train_loss, torch.Tensor) else avg_epoch_train_loss

            #7. 计算验证集、测试集损失，并打印，根据验证集损失确定是否早停
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 初始化 stop_signal 以供所有进程使用
            stop_signal = torch.tensor(0.0).to(self.device)


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            epoch_time = time.time() # 更新 epoch_time
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, avg_epoch_train_loss, vali_loss, test_loss))

            # 主进程判断早停，若满足条件，计数并保存权重
            early_stopping(vali_loss, self.model.module if self.args.ddp else self.model, self.path)

            # 验证停止信号，设置停止信号（因为是在主进程循环，所以这里只设置了主进程的停止信号）
            if early_stopping.early_stop:
                print("Early stopping")
                stop_signal = torch.tensor(1.0).to(self.device)

            if self.args.ddp:   # 设置其他设备的停止信号为0，广播主进程的停止信号到其他设备，若满足条件，break
                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() == 1.0:
                    dist.barrier()
                    break

            #9. TODO amp精度和累计梯度 清理gpu内存
            torch.cuda.empty_cache()

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss_sum = torch.tensor(0.0).to(self.device)
        total_samples = 0       #记录总样本数以计算加权平均

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                
                # encoder - decoder
                outputs, target_y = self._forward_pass(batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark)

                # 直接在 GPU 上计算损失
                loss = criterion(outputs, target_y)

                batch_size = target_y.shape[0]
                total_loss_sum += loss.item() * batch_size
                total_samples += batch_size

        # Synchronize across processes if DDP is enabled
        if self.args.ddp:
            # Sum the total loss and total samples across all GPUs
            dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = total_samples_tensor.item()

        avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0

        # Return the average loss (as a float or tensor, consistency is key)
        return avg_loss.item() if isinstance(avg_loss, torch.Tensor) else float(avg_loss)

    def load_ddp_model(self,setting):
        best_model_path = os.path.join(self.args.checkpoints, setting) + '/' + 'checkpoint.pth'
        self.model = self._build_model() # 先构建模型
        map_location = self.device if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(best_model_path, map_location=map_location)

        
        # 创建一个新的字典来存储可能修改后的键值对
        is_ddp_state_dict = all(key.startswith('module.') for key in state_dict.keys())
        new_state_dict = {} 
        if is_ddp_state_dict:
            for k, v in state_dict.items():
                name = k[len('module.'):] # 移除 'module.' 前缀
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict # 直接使用，load_state_dict 默认忽略不匹配的键
        
        #加载模型
        self.model.load_state_dict(new_state_dict, strict=True) 
        print(f"Successfully loaded model checkpoint from {best_model_path}")
        
        return self.model
    
    def test(self, setting):
        #1. 加载数据
        _, test_loader = self._get_data(flag='test')

        #2. 加载最优模型
        self.model = self.load_ddp_model(setting)

        #3. 准备计算acc的变量
        preds = []
        trues = []
        inputx = []
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #4. 模型切换到eval
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs, batch_y = self._forward_pass(batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        # if self.args.test_flop:
        #     test_params_flop((batch_x.shape[1],batch_x.shape[2]))
        #     exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        
        # result save,覆盖之前的folder_path，用于存储pred
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
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
        pass
    #     pred_data, pred_loader = self._get_data(flag='pred')

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         if self.args.use_multi_gpu and self.args.use_gpu and dist.is_initialized():
    #             # 分布式环境中加载模型
    #             map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
    #             self.model.module.load_state_dict(torch.load(best_model_path, map_location=map_location))
    #         else:
    #             self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
    #             # 使用封装的前向传播方法
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs, target_y = self._forward_pass(batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 outputs, target_y = self._forward_pass(batch_x, batch_y, batch_x_mark, dec_inp, batch_y_mark)
                    
    #             pred = outputs.detach().cpu().numpy()
    #             preds.append(pred)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #     if self.is_main_process:
    #         np.save(folder_path + 'real_prediction.npy', preds)

    #     return
