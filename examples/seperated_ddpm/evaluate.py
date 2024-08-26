from diffusers import SeperatedDDPMScheduler as DDPMScheduler
from diffusers import UNet2DModel
from datasets import load_dataset

import numpy as np

import torch
from torchvision import utils
from torch.utils.data import DataLoader

import argparse
import os
from tqdm import tqdm

from process_dataset import create_conditional_subdatasets, create_unconditional_subdatasets

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from evaluate_util import FIDWithFeatures, compute_fid, compute_prdcf1, replicate_test

def generate_images(model,
                    scheduler,
                    batch_size, 
                    num_channel, 
                    sample_size,
                    mu,
                    target_noise_mean_index):
    
    noise = torch.randn((batch_size, num_channel, sample_size, sample_size)).to(model.device)
    noise = noise + mu
    
    x = noise
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps):
            if t == 0: break
            model_output = model(x, t).sample
            x = scheduler.step(model_output, t, x, target_noise_mean_index=target_noise_mean_index).prev_sample
            
    return x

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
        "--num_subsets",
        type=int,
        default=2,
        help=("Number of subsets for training. (only for unconditional)"),
    )
parser.add_argument(
        "--samples_per_train_subset",
        type=int,
        default=1024,
        help=("Number of samples of each subset used for training. (only for unconditional)"),
    )
parser.add_argument(
        "--samples_per_test_subset",
        type=int,
        default=1024,
        help=("Number of samples of each subset for test set. (only for unconditional)"),
    )
parser.add_argument(
        "--fid_batch_size", type=int, default=256, help="The batch size for evaluation."
    )
parser.add_argument(
        "--fid_total_size", type=int, default=1024, help="The number of total images to generate for evaluation."
    )
parser.add_argument(
        "--target_noise_means_input",
        type=str,
        default="",
        help="Comma-separated noise mean values for each class, e.g., '-1,1' for two classes.",
    )
parser.add_argument(
        "--target_noise_std",
        type=float,
        default=1.,
        help=("Standard Deviation of the latent noise image at time step T"),
    )
parser.add_argument(
        "--condition_type",
        type=str,
        default="unconditional",
        choices=["conditional", "unconditional"],
        help="Conditional type seperates target noise means by class, whereas unconditional type seperate target noise means by subset with equally distributed classes",
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
target_noise_means = [float(x) for x in args.target_noise_means_input.split(',')]
num_subsets = len(target_noise_means) 

repeat_count_fake = args.fid_total_size // args.fid_batch_size
residue = args.fid_total_size % args.fid_batch_size
assert(residue == 0)

n_real_train = args.samples_per_train_subset * num_subsets
n_real_test = args.samples_per_test_subset * num_subsets
n_fake = args.fid_total_size * num_subsets

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

if args.condition_type == 'conditional':    
    train_subsets = create_conditional_subdatasets(train_set, args, class_labels, args.samples_per_class_train)
    test_subsets = create_conditional_subdatasets(test_set, args, class_labels, args.samples_per_class_test)
else:
    train_subsets = create_unconditional_subdatasets(train_set, args, class_labels, num_subsets, args.samples_per_train_subset)
    test_subsets = create_unconditional_subdatasets(test_set, args, class_labels, num_subsets, args.samples_per_test_subset)
    
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

real_images_train = []
real_images_test = []
fake_images = []

for (mu, subset) in zip(target_noise_means, train_subsets):
    # Get Real Images
    dataloader = DataLoader(subset, len(subset))
    for x, labels in dataloader:
        filename = output_dir + f'/train/real_images_mean{mu}.png'
        x_processed = (x / 2 + 0.5).clamp(0, 1)
        utils.save_image(x_processed[:10], filename, nrow=10)
        real_images_train.append(x_processed)
        break  # We only need the first batch, so break the loop
real_images_train = torch.cat(real_images_train, dim=0)

assert(len(real_images_train) == n_real_train)
    
for (mu, subset) in zip(target_noise_means, test_subsets):
    # Get Real Images
    dataloader = DataLoader(subset, len(subset))
    for x, labels in dataloader:
        filename = output_dir + f'/test/real_images_mean{mu}.png'
        x_processed = (x / 2 + 0.5).clamp(0, 1)
        utils.save_image(x_processed[:10], filename, nrow=10)
        real_images_test.append(x_processed)
        break  # We only need the first batch, so break the loop    
real_images_test = torch.cat(real_images_test, dim=0)

assert(len(real_images_test) == n_real_test)

for i, mu in enumerate(target_noise_means):
    #
    # Generate Fake Images
    #
    fake_batch = []
    for n in range(repeat_count_fake):
        print(f"Running Reverse Process... | subset: {i+1} / {len(target_noise_means)} | batch: {n+1} / {repeat_count_fake}")
        x = generate_images(model, scheduler, args.fid_batch_size, num_channel, sample_size, mu, target_noise_mean_index=i)
        x_processed = (x / 2 + 0.5).clamp(0, 1)
        fake_batch.append(x_processed)
        
    fake_batch = torch.cat(fake_batch, dim=0)
    filename = output_dir + f'/fake_images_mean{mu}.png'
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
