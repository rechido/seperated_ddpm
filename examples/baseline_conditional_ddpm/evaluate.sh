DATASET="/nas/dataset/daeun/raw_images/cifar10-32-perclass3000-30000/"
OUTPUT_DIR='20240805_cifar10-32-perclass3000-30000_10000epoch'

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset_name=$DATASET \
  --resolution=32 \
  --output_dir=$OUTPUT_DIR \
  --fid_batch_size=100 \
  --fid_total_size=3000 \
  --target_classes="0,1,2,3,4,5,6,7,8,9" \
  --samples_per_class_train=3000 \
  --samples_per_class_test=2000 \
  --prdc_batch_size=4 \
  --num_checkpoint=13125000