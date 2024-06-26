if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/EcmP_mk3/ACL18_2016" ]; then
    mkdir ./logs/EcmP_mk3/ACL18_2016
fi

model_name=EcmP_mk3

#patching setting
first_stage_patching=None
second_stage_patching=linear
label_len=4 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)

#decomposition
decomposition=1
kernel_size=9

#extras
result_log_path=./result_log/EcmP_mk3/acl_18_ext/acl_2016_v5.txt

root_path_name=./data/ACL_18_EXT/2016/
#data_path_name=stock_000001.SZ.csv
#model_id_name=stock_000001SZ
data_name=stock_custom

random_seed=2023

dt_format_str=0

target=target

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
            --enc_in 6 \
            --e_layers 2 \
            --n_heads 4 \
            --d_patch 0 \
            --d_model 64 \
            --d_ff 128 \
            --dropout 0.1\
            --fc_dropout 0.1\
            --head_dropout 0\
            --patch_len 1\
            --stride 1\
            --des 'Exp' \
            --train_epochs 50\
            --patience 20\
            --lradj 'TST'\
            --pct_start 0.06\
            --itr 1 --batch_size 8 --learning_rate 0.00001 >logs/EcmP_mk3/ACL18_2016/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
        done

    fi

done