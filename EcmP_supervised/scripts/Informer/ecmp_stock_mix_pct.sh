if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Informer" ]; then
    mkdir ./logs/Informer
fi

model_name=Informer


label_len=4 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)


#extras
result_log_path=./result_log/Informer/ecmp_stock_mix_pct.txt

root_path_name=./data/EcmP_stock_L_2016_24_mix/
data_name=stock_custom

random_seed=2023

dt_format_str=0

target=close_pct_change

scale=1


break_resume=1 #1 means start from fresh, 0 means start from the resume file
resume_file=None

for full_path_n in $root_path_name*.csv
do
    data_path_name=$(basename $full_path_n)
    model_id_name="${data_path_name%.*}"

    if [ "$data_path_name" = "$resume_file" ] || [ $break_resume -eq 1 ]; then

        if [ "$data_path_name" = "$resume_file" ] && [ $break_resume -eq 0 ]; then
            echo "found the break resume file, it is $data_path_name"
            break_resume=1
        fi


        for pred_len in 10 20 40 60
        do
            seq_len=40
            python -u EcmP_supervised/run_longExp.py \
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
            --dec_in 9 \
            --c_out 1 \
            --e_layers 2 \
            --d_layers 1 \
            --n_heads 4 \
            --d_model 32 \
            --d_ff 64 \
            --des 'Exp' \
            --train_epochs 50\
            --patience 19\
            --itr 1 --batch_size 16 --learning_rate 0.0005 >logs/Informer/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
        done

    fi

done