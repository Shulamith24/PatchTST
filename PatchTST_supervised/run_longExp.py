import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.ddp import is_main_process, init_distributed_mode



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # DDP
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='local rank passed from distributed launcher')
    parser.add_argument('--debug', action='store_true',default=False, help='调试模式，不使用ddp')
    parser.add_argument('--acc_steps', type=int, default=1, help='DDP等价于更大的batch=batch_size*acc_steps')
    parser.add_argument('--ddp', action='store_true', default=False, help='使用ddp')
    
if __name__ == '__main__':
    args = parser.parse_args()


    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    #分布式训练初始化
    args.ddp = True if (not args.debug) and args.use_multi_gpu else False
    init_distributed_mode(args)

    # 记录主进程标识符，因为 args.ddp 可能会在后面被修改
    is_distributed = args.ddp 
    is_main = is_main_process() # 存储初始的主进程状态

    try:
        print('Args in experiment:')
        print(args)

        Exp = Exp_Main

        if args.is_training:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des)

            exp = Exp(args,setting)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train()
            torch.cuda.empty_cache()

            # --- 训练结束，准备测试 ---
            if is_distributed and torch.distributed.is_initialized():
                torch.distributed.barrier() 
                torch.distributed.destroy_process_group()
                print("DDP process group destroyed after training.")
            
            # --- 仅在原主进程上执行测试 ---
            if is_main: 
                args_test = argparse.Namespace(**vars(args)) # 创建 args 的副本以进行修改
                args_test.ddp = False
                args_test.use_multi_gpu = False
                args_test.gpu = 0 # 或者指定一个用于测试的 GPU ID
                args_test.device = torch.device('cuda:{}'.format(args_test.gpu) if torch.cuda.is_available() else 'cpu')
                
                # 使用更新后的 args 创建新的 Exp_Main 实例进行测试
                exp_test = Exp(args_test, setting) 
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp_test.test(setting) 

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp_test.predict(setting, load=True) # load=True 确保加载模型

                torch.cuda.empty_cache()


        else: # 如果 args.is_training 为 False，直接执行测试
             if is_main: # 非训练模式也只在主进程执行测试
                ii = 0
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}'.format(args.model_id,
                                                                                                            args.model,
                                                                                                            args.data,
                                                                                                            args.features,
                                                                                                            args.seq_len,
                                                                                                            args.label_len,
                                                                                                            args.pred_len,
                                                                                                            args.d_model,
                                                                                                            args.n_heads,
                                                                                                            args.e_layers,
                                                                                                            args.d_layers,
                                                                                                            args.d_ff,
                                                                                                            args.factor,
                                                                                                            args.embed,
                                                                                                            args.distil,
                                                                                                            args.des)
                # 确保测试时 args.ddp 为 False
                args.ddp = False
                args.use_multi_gpu = False
                args.local_rank = -1
                
                exp = Exp(args, setting)
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
            # else: 非主进程在 is_training=False 时不执行任何操作
    finally:
        # 如果手动终止，清理ddp进程组
        print("Script execution finished.")
        # 如果 DDP 因为异常提前终止，可能仍需清理
        if is_distributed and torch.distributed.is_initialized():
             torch.distributed.destroy_process_group()
             if is_main:
                  print("DDP process group potentially destroyed again in finally block (e.g., due to error).")