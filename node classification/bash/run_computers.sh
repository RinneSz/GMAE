[ -z "${exp_name}" ] && exp_name="computers"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--peak_lr 1e-4 --end_lr 1e-9 --l1 5 --l2 5 --batch_size 128 --ffn_dim 64 --hidden_dim 64 --num_heads 8 --dropout_rate 0.5"
[ -z "${mask_ratio}" ] && mask_ratio="0.8"
[ -z "${warmup_updates}" ] && warmup_updates="40000"
[ -z "${tot_updates}" ] && tot_updates="400000"
[ -z "${n_encoder_layers}"] && n_encoder_layers="30"
[ -z "${n_decoder_layers}"] && n_decoder_layers="2"
[ -z "${dataset_name}"] && dataset_name="Amazon-Computers"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "exp_name: ${exp_name}"
echo "warmup_updates: ${warmup_updates}"
echo "tot_updates: ${tot_updates}"
echo "n_encoder_layers: ${n_encoder_layers}"
echo "n_decoder_layers: ${n_decoder_layers}"
echo "==============================================================================="

save_path="exps/$exp_name-$n_encoder_layers-$n_decoder_layers-$mask_ratio/$seed"
mkdir -p $save_path

python entry.py --num_workers 8 --seed $seed \
      --mask_ratio $mask_ratio --dataset_name $dataset_name \
      --gpus 1 --accelerator ddp --precision 16 \
      $arch --n_encoder_layers $n_encoder_layers --n_decoder_layers $n_decoder_layers \
      --warmup_updates $warmup_updates --tot_updates $tot_updates \
      --default_root_dir $save_path --reload_dataloaders_every_epoch 1
