#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# 设置GPU可见设备
export CUDA_VISIBLE_DEVICES=3,4,5 # 根据实际可用GPU数量调整
export OMP_NUM_THREADS=4
seq_len=336
model_name=PatchTST

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    torchrun --nproc_per_node=3 \
            --master_addr="localhost" \
            --master_port=$(shuf -i 20000-30000 -n 1) \
        run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 100 \
      --patience 10 \
      --num_workers 0 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.0002 \
      --use_gpu True \
      --use_multi_gpu
done