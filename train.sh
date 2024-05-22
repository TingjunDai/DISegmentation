method=${1:-"BiRefNet"} # 训练模型
epochs=10
val_last=10
step=1
# Train
echo Training started at $(date)
python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
                --val_last ${val_last} --step ${step}
echo Training finished at $(date)