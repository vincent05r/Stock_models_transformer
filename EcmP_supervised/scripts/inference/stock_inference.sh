#inference data

data_name=stock_custom_pred
root_path_name=
full_path_n=
data_path_name=$(basename $full_path_n)
model_id_name="${data_path_name%.*}"

#loading path
prev_scaler=./scaler/INTC_pct_index/INTC_pct_index_50_1.pkl
pred_model_load_path=./checkpoints/INTC_pct_index_50_1_PatchTST_stock_custom_ftMS_sl50_ll3_pl1_dm64_nh8_el2_dl1_df128_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth


#consistent with model

model_name=EcmP_mk3

#patching setting
first_stage_patching=MLP
second_stage_patching=None

#decomposition
decomposition=0
kernel_size=9


dt_format_str=0

target=close_pct_change

scale=1


seq_len=50 #match the .pth hyperparameters
label_len=3 #reminder the label length is different to the predicted length, lead time(overlap) time between x(input) and y(label)
pred_len=1


python -u EcmP_supervised/run_inference.py \
--prev_scaler $prev_scaler \
--pred_model_load_path $pred_model_load_path \
--decomposition $decomposition\
--kernel_size $kernel_size\
--first_stage_patching $first_stage_patching\
--second_stage_patching $second_stage_patching\
--root_path $root_path_name \
--data_path $data_path_name \
--model_id 'PTFT2016V2PCIE_'${data_path_name%.csv}'_'$seq_len'_'$pred_len \
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
--d_model 108 \
--d_ff 256 \
--dropout 0\
--fc_dropout 0\
--head_dropout 0\
--patch_len 5\
--stride 1\