#inference data

data_name=stock_custom_pred
root_path_name=./data/inference/
full_path_n=./data/inference/amd_hft_test.csv
data_path_name=$(basename $full_path_n)
model_id_name="${data_path_name%.*}"

#loading path
pred_model_load_path=./finetune_cp/HFT_FT_2411m_A_pct_120_10_EcmP_mk3_stock_custom_ftMS_sl120_ll0_pl10_dm108_dp0_pl10_nh6_el3_dl1_df256_fc1_ebtimeF_Exp_dcomp0_kn9_LOlinears_None_rv1_close_pct_change/checkpoint.pth


#consistent with model

model_name=EcmP_mk3

#patching setting
first_stage_patching=LOlinears
second_stage_patching=None

#decomposition
decomposition=0
kernel_size=9


dt_format_str=0

target=close_pct_change

scale=0
revin=1

seq_len=120 #match the .pth hyperparameters
label_len=0 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)
pred_len=10


python -u EcmP_supervised/run_inference.py \
--pred_model_load_path $pred_model_load_path \
--decomposition $decomposition \
--kernel_size $kernel_size \
--first_stage_patching $first_stage_patching \
--second_stage_patching $second_stage_patching \
--root_path $root_path_name \
--data_path $data_path_name \
--model_id 'IF_'${data_path_name%.csv} \
--model $model_name \
--data $data_name \
--features MS \
--seq_len $seq_len \
--label_len $label_len \
--pred_len $pred_len \
--scale $scale \
--revin $revin \
--target $target \
--dt_format_str $dt_format_str \
--enc_in 9 \
--e_layers 3 \
--n_heads 6 \
--d_patch 0 \
--d_model 108 \
--d_ff 256 \
--dropout 0 \
--fc_dropout 0 \
--head_dropout 0 \
--patch_len 10 \
--stride 9