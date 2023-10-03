if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=100
label_len=2 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)
model_name=PatchTST

result_log_path=./result_log/result_sse.txt

root_path_name=./data/stock_benchmark/
#data_path_name=stock_000001.SZ.csv
#model_id_name=stock_000001SZ
data_name=stock_custom

random_seed=2023

dt_format_str=0

target=close

scale=True

full_path_n=./data/stock_benchmark/sse_index_spec_s.csv

data_path_name=$(basename $full_path_n)
model_id_name="${data_path_name%.*}"


for seq_len in 7 15 30 50 100 200
do
    for pred_len in 1
    do
        python3.9 -u PatchTST_supervised/run_longExp.py \
        --result_log_path $result_log_path\
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
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
        --e_layers 2 \
        --n_heads 4 \
        --d_model 64 \
        --d_ff 128 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 8\
        --stride 4\
        --des 'Exp' \
        --train_epochs 100\
        --patience 10\
        --lradj 'TST'\
        --pct_start 0.2\
        --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done