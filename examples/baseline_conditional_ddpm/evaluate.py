from diffusers import DDPMScheduler, UNet2DModel
from datasets import load_dataset

import numpy as np
from torch.utils.data.sampler import RandomSampler as RandomSampler

import torch
from torchvision import utils
from torchvision import transforms

import argparse
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from evaluate_util import FIDWithFeatures, compute_fid, compute_prdcf1, replicate_test

#
# Input
#
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
parser.add_argument("--ddpm_num_steps", type=int, default=1000)
parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
parser.add_argument(
        "--target_classes",
        type=str,
        default="",
        help="Comma-separated class numbers to train, e.g., '4,5' for 2-class MNIST.",
    )
parser.add_argument(
        "--samples_per_class_train",
        type=int,
        default=1024,
        help=("Number of samples of each class used for training. (only for conditional)"),
    )
parser.add_argument(
        "--samples_per_class_test",
        type=int,
        default=1024,
        help=("Number of samples of each class for test set. (only for conditional)"),
    )
parser.add_argument(
        "--fid_batch_size", type=int, default=256, help="The batch size for evaluation."
    )
parser.add_argument(
        "--fid_total_size", type=int, default=1024, help="The number of total images to generate for evaluation."
    )
parser.add_argument(
        "--prdc_batch_size", type=int, default=4, help="The batch size for evaluation."
    )
parser.add_argument(
        "--num_checkpoint", type=int, default=-1, help="Iteration number of the checkpoint model"
    )

args = parser.parse_args()

device = 'cuda:0'
class_labels = [int(x) for x in args.target_classes.split(',')]
num_class = len(class_labels)

repeat_count_fake = args.fid_total_size // args.fid_batch_size
residue = args.fid_total_size % args.fid_batch_size
assert(residue == 0)

n_real_train = args.samples_per_class_train * num_class
n_real_test = args.samples_per_class_test * num_class
n_fake = args.fid_total_size * num_class

output_dir = args.output_dir
if args.num_checkpoint > 0:
    output_dir += f'/checkpoint-{args.num_checkpoint}'
output_dir += f'/evaluation_real_train{n_real_train}_real_test{n_real_test}_fake{n_fake}'
print(output_dir)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + '/train/', exist_ok=True)
os.makedirs(output_dir + '/test/', exist_ok=True)



#
# Load Data
#
train_set = load_dataset(args.dataset_name + 'train/', split='train')
test_set = load_dataset(args.dataset_name + 'test/', split='train')

augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def transform_images(examples):
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}

train_set.set_transform(transform_images)
test_set.set_transform(transform_images)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)

real_batch = []
for batch in train_dataloader:
    x = batch["input"]
    x_processed = (x / 2 + 0.5).clamp(0, 1)
    real_batch.append(x_processed)
    filename = output_dir + f'/real_images_train.png'
    utils.save_image(x_processed[:10], filename, nrow=10)
real_images_train = torch.cat(real_batch, dim=0).to(device)

assert(len(real_images_train) == n_real_train)

real_batch = []
for batch in test_dataloader:
    x = batch["input"]
    x_processed = (x / 2 + 0.5).clamp(0, 1)
    real_batch.append(x_processed)
    filename = output_dir + f'/real_images_test.png'
    utils.save_image(x_processed[:10], filename, nrow=10)
real_images_test = torch.cat(real_batch, dim=0).to(device)

assert(len(real_images_test) == n_real_test)



#
# Load the final model
#
if args.num_checkpoint > 0:
    model = UNet2DModel.from_pretrained(f'{args.output_dir}/checkpoint-{args.num_checkpoint}', subfolder="unet").to(device)
else:
    model = UNet2DModel.from_pretrained(args.output_dir, subfolder="unet").to(device)
scheduler = DDPMScheduler.from_pretrained(args.output_dir + '/scheduler' )
scheduler.set_timesteps(args.ddpm_num_inference_steps)
#
# Create an input noisy image (x)
#
sample_size = model.config.sample_size # H, W resolution
num_channel = model.config.in_channels # channel

fake_images = []

for i, c in enumerate(class_labels):
    print(f'Generate fake images of class {c}...')
    
    fake_batch = []
    for n in range(repeat_count_fake):
        print(f"Running Reverse Process... | class: {i+1} / {len(class_labels)} | batch: {n+1} / {repeat_count_fake}")
        noise = torch.randn((args.fid_batch_size, num_channel, sample_size, sample_size)).to("cuda")
        x = noise
        label = torch.ones((args.fid_batch_size), dtype=torch.int64).to("cuda") * c
        
        # inference
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():
                noisy_residual = model(x, t, label).sample
                prev_noisy_sample = scheduler.step(noisy_residual, t, x).prev_sample
                x = prev_noisy_sample
                
        x_processed = (x / 2 + 0.5).clamp(0, 1)
        fake_batch.append(x_processed)
    
    fake_batch = torch.cat(fake_batch, dim=0)
    filename = output_dir + f'/fake_images_class{c}.png'
    utils.save_image(fake_batch[:10], filename, nrow=10)
    fake_images.append(fake_batch)
    
fake_images = torch.cat(fake_images, dim=0)

print('real_train: ', torch.mean(real_images_train).item(), torch.std(real_images_train).item(), torch.min(real_images_train).item(), torch.max(real_images_train).item())
print('real_test: ', torch.mean(real_images_test).item(), torch.std(real_images_test).item(), torch.min(real_images_test).item(), torch.max(real_images_test).item())
print('fake:', torch.mean(fake_images).item(), torch.std(fake_images).item(), torch.min(fake_images).item(), torch.max(fake_images).item())

print(f'real_train={len(real_images_train)}, real_test={len(real_images_test)}, fake={len(fake_images)}')

assert(len(fake_images) == n_fake)

#
# evaluation for train set
#
fid_metric = FIDWithFeatures(device=device)
real_features, fake_features = compute_fid(real_images_train, fake_images, fid_metric, output_dir, split='train')

k_set=[1,2,3,4,5,6,7,8,9,10]
compute_prdcf1(real_images_train, fake_images, k_set, output_dir, distance_metric='cosine_similarity', datatype='image', split='train')
compute_prdcf1(real_features, fake_features, k_set, output_dir, distance_metric='euclidean', datatype='feature', split='train')

weight_set = [1,2,3,4,5,6,7,8,9] # [3] is recommended
thresholds = np.linspace(0, 1e-1, 10001)
replicate_test(real_images_train, fake_images, weight_set, thresholds, output_dir, distance_metric='cosine_similarity', datatype='image', split='train')
thresholds = np.linspace(0, 100, 10001)
replicate_test(real_features, fake_features, weight_set, thresholds, output_dir, distance_metric='euclidean', datatype='feature', split='train', real_images=real_images_train, fake_images=fake_images)

#
# evaluation for test set
#
fid_metric = FIDWithFeatures(device=device)
real_features, fake_features = compute_fid(real_images_test, fake_images, fid_metric, output_dir, split='test')

k_set=[1,2,3,4,5,6,7,8,9,10]
compute_prdcf1(real_images_test, fake_images, k_set, output_dir, distance_metric='cosine_similarity', datatype='image', split='test')
compute_prdcf1(real_features, fake_features, k_set, output_dir, distance_metric='euclidean', datatype='feature', split='test')

weight_set = [1,2,3,4,5,6,7,8,9] # [3] is recommended
thresholds = np.linspace(0, 1e-1, 10001)
replicate_test(real_images_test, fake_images, weight_set, thresholds, output_dir, distance_metric='cosine_similarity', datatype='image', split='test')
thresholds = np.linspace(0, 100, 10001)
replicate_test(real_features, fake_features, weight_set, thresholds, output_dir, distance_metric='euclidean', datatype='feature', split='test', real_images=real_images_test, fake_images=fake_images)

print('done')
