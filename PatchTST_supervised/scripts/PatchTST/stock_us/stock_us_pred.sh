seq_len=50 #match the .pth hyperparameters
label_len=3 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)
pred_len=1
model_name=PatchTST

result_log_path=./result_log/result_spec1.txt

root_path_name=./data/stock_us_pred/
#data_path_name=stock_000001.SZ.csv
#model_id_name=stock_000001SZ
data_name=stock_custom_pred

random_seed=2023

scale=False
pred_model_load_path=./checkpoints/BA_pct_index_50_1_PatchTST_stock_custom_ftMS_sl50_ll3_pl1_dm64_nh8_el3_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth

full_path_n=./data/stock_us_pred/test_1.csv

data_path_name=$(basename $full_path_n)
model_id_name="${data_path_name%.*}"

target=close_pct_change


python3.9 -u PatchTST_supervised/run_longExp.py \
--do_predict True \
--is_training 0 \
--root_path $root_path_name \
--data_path $data_path_name \
--model $model_name \
--data $data_name \
--features MS \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--target $target \
--scale $scale \
--pred_model_load_path $pred_model_load_path \
--enc_in 10 \
--e_layers 3 \
--n_heads 8 \
--d_model 64 \
--d_ff 128 \
--dropout 0.2 \
--fc_dropout 0.2 \
--head_dropout 0 \
--patch_len 8 \
--stride 4 \
--des 'Exp' \
--train_epochs 100 \
--patience 10 \
--lradj 'TST' \
--pct_start 0.2 \
--itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 