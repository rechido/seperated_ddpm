

DEVICE=0

dataset_name='/nas/dataset/daeun/raw_images/cifar10-32-perclass128-256/train/'

output_dir='./20240729_cifar10-32-perclass128-256_10000epoch'

CUDA_VISIBLE_DEVICES=$DEVICE python3 train_conditional.py \
  --dataset_name=$dataset_name \
  --resolution=32 \
  --output_dir=$output_dir \
  --resume_from_checkpoint 'latest' \
  --train_batch_size=16 \
  --num_epochs=10000 \
  --save_model_epochs=1000 \
  --checkpointing_steps=-1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no 

