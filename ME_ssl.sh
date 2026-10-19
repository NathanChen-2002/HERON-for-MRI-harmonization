
export PYTHONHASHSEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  main.py \
  --save_dir "" \
  --experiment_description "" \
  --run_description "" \
  --harmonizing_model HERON \
  --ssl_method MultiscalePretraining \
  --train_mode ssl \
  --branch2ssl multiscale \
  --dataset ssl_dataset \
  --seed 0 \
  --data_path "" \
  --hparam_overrides ""
 