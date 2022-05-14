[ -z "${exp_name}" ] && exp_name="mutag"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--early_stop_epoch 50 --peak_lr 1e-4 --end_lr 1e-9 --ffn_dim 80 --hidden_dim 80 --num_heads 8 --dropout_rate 0.1 --edge_type multi_hop --multi_hop_max_dist 20"
[ -z "${mask_ratio}"] && mask_ratio="0.9"
[ -z "${warmup_updates}" ] && warmup_updates="40000"
[ -z "${tot_updates}" ] && tot_updates="400000"
[ -z "${n_encoder_layers}"] && n_encoder_layers="1"
[ -z "${n_decoder_layers}"] && n_decoder_layers="2"
[ -z "${dataset_name}"] && dataset_name="MUTAG"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "mask_ratio: ${mask_ratio}"
echo "n_encoder_layers: ${n_encoder_layers}"
echo "n_decoder_layers: ${n_decoder_layers}"
echo "==============================================================================="

save_path="exps/$exp_name-$n_encoder_layers-$n_decoder_layers-$mask_ratio/$seed"
mkdir -p $save_path

python entry.py --num_workers 8 --seed $seed --batch_size 16 \
      --dataset_name $dataset_name \
      --gpus 1 --accelerator ddp --precision 16 \
      $arch --n_encoder_layers $n_encoder_layers --n_decoder_layers $n_decoder_layers\
      --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir $save_path --mask_ratio $mask_ratio
