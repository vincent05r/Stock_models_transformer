if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
label_len=50 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)
model_name=PatchTST

root_path_name=./PatchTST_supervised/stock_data/
#data_path_name=stock_000001.SZ.csv
#model_id_name=stock_000001SZ
data_name=stock_custom

random_seed=2023

break_resume=0 #1 means start from fresh, 0 means start from the resume file
resume_file=stock_000559SZ.csv

for full_path_n in $root_path_name*.csv
do
    data_path_name=$(basename $full_path_n)
    model_id_name="${data_path_name%.*}"

    if [ "$data_path_name" = "$resume_file" ] || [ $break_resume -eq 1 ]; then

        if [ "$data_path_name" = "$resume_file" ] && [ $break_resume -eq 0 ]; then
            echo "found the break resume file, it is $data_path_name"
            break_resume=1
        fi
        

        for seq_len in 50 100 200
        do
            for pred_len in 1 7 14 30
            do
                python3.9 -u PatchTST_supervised/run_longExp.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path $root_path_name \
                --data_path $data_path_name \
                --model_id $model_id_name'_'$seq_len'_'$pred_len \
                --model $model_name \
                --data $data_name \
                --features MS \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --enc_in 321 \
                --e_layers 3 \
                --n_heads 16 \
                --d_model 128 \
                --d_ff 256 \
                --dropout 0.2\
                --fc_dropout 0.2\
                --head_dropout 0\
                --patch_len 16\
                --stride 8\
                --des 'Exp' \
                --train_epochs 100\
                --patience 10\
                --lradj 'TST'\
                --pct_start 0.2\
                --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
            done
        done

    fi

done