if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/EcmP_mk2" ]; then
    mkdir ./logs/EcmP_mk2
fi
seq_len=336
model_name=EcmP_mk2

#extras
result_log_path=./result_log/EcmP_mk2/weather.txt
#mk2 setting
dcomp_individual=0

root_path_name=./data/ts_benchmark
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u EcmP_supervised/run_longExp.py \
      --dcomp_individual $dcomp_individual \
      --result_log_path $result_log_path \
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
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 12 \
      --d_model 168 \
      --d_ff 168 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --itr 1 --batch_size 128 --learning_rate 0.00001 >logs/EcmP_mk2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done