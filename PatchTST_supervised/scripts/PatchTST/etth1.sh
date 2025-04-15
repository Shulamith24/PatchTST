if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=PatchTST

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=3,4  # 指定使用哪些GPU，python对这三个gpu重新排序，3->0
export OMP_NUM_THREADS=4

random_seed=2021
for pred_len in 96 192 336 720
do
    torchrun --nproc_per_node=$NUM_GPUS \
            --nnodes=1 \
            --master_addr="localhost" \
            --master_port=$(shuf -i 20000-30000 -n 1) \
            run_longExp.py \
      --use_multi_gpu \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 8\
      --stride 8\
      --num_workers 0\
      --des 'Exp' \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 --batch_size 32 --learning_rate 0.0001
done