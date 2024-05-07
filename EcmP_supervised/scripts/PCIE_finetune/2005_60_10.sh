if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


if [ ! -d "./logs/PCIE" ]; then
    mkdir ./logs/PCIE
fi

if [ ! -d "./logs/PCIE/finetune" ]; then
    mkdir ./logs/PCIE/finetune
fi

model_name=EcmP_mk3

#patching setting
first_stage_patching=MLP
second_stage_patching=None
label_len=0 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)

#decomposition
decomposition=0
kernel_size=9


data_name=stock_custom
root_path_name=./data/EcmP_stock_L_2005_24/
data_path_name=AAPL_pct.csv

random_seed=2023

dt_format_str=0

target=close

scale=1


for pred_len in 10
do
    seq_len=60
    python -u EcmP_supervised/run_pretrain.py \
    --is_pretrain 1\
    --pe zeros\
    --learn_pe True\
    --decomposition $decomposition\
    --kernel_size $kernel_size\
    --first_stage_patching $first_stage_patching\
    --second_stage_patching $second_stage_patching\
    --random_seed $random_seed \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id 'PT2005V2PCIE_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --scale $scale\
    --target $target\
    --dt_format_str $dt_format_str\
    --enc_in 9 \
    --e_layers 3 \
    --n_heads 3 \
    --d_patch 0 \
    --d_model 45 \
    --d_ff 128 \
    --dropout 0.1\
    --fc_dropout 0.1\
    --head_dropout 0\
    --patch_len 4\
    --stride 1\
    --des 'Exp' \
    --train_epochs 10\
    --lradj 'TST'\
    --pct_start 0.1\
    --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/PCIE/pretrain/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done