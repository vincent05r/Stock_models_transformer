if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/EcmP_mk3/ecmp_stock" ]; then
    mkdir ./logs/EcmP_mk3/ecmp_stock
fi

model_name=EcmP_mk3

#patching setting
first_stage_patching=linear
second_stage_patching=None
label_len=0 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)

#decomposition
decomposition=0
kernel_size=9

#extras
result_log_path=./result_log/EcmP_mk3/ecmp_stock_pct/final_10_40_pct_xll_s40_revin.txt

root_path_name=./data/EcmP_stock_L_2016_24_spec_official/
#data_path_name=stock_000001.SZ.csv
#model_id_name=stock_000001SZ
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


        for pred_len in 10 20 40
        do
            seq_len=40
            python -u EcmP_supervised/run_longExp.py \
            --revin 0\
            --decomposition $decomposition\
            --kernel_size $kernel_size\
            --first_stage_patching $first_stage_patching\
            --second_stage_patching $second_stage_patching\
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
            --enc_in 5 \
            --e_layers 2 \
            --n_heads 5 \
            --d_patch 0 \
            --d_model 40 \
            --d_ff 64 \
            --dropout 0.1\
            --fc_dropout 0.1\
            --head_dropout 0\
            --patch_len 4\
            --stride 1\
            --des 'Exp' \
            --train_epochs 50\
            --patience 19\
            --lradj 'TST'\
            --pct_start 0.1\
            --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/EcmP_mk3/ecmp_stock/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
        done

    fi

done