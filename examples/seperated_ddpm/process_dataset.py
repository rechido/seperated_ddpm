import torch
from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from torchvision import transforms

import random

def process_dataset(dataset, config, class_indices):
    
    # Preprocessing the datasets and DataLoaders creation.
    transform_list = [
                transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
                transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                #transforms.Normalize([0.5], [0.5]),
                transforms.Normalize([0.1307], [0.3081]),
            ]
    
    def transform_images(examples):
        augmentations = transforms.Compose(transform_list)
        if 'cifar10' == config.dataset_name:    images = [augmentations(image.convert("RGB")) for image in examples["img"]]
        else:                                   images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        labels = [label for label in examples["label"]]
        return {"input": images, "labels": labels}
    
    filtered_dataset = dataset.filter(lambda example: example['label'] in class_indices)
    filtered_dataset.set_transform(transform_images)
    
    return filtered_dataset


# Each Subdataset contain the data with only one class.
class ConditionalSubdataset(Dataset):
    def __init__(self, dataset, config, class_label, n_samples):
        self.dataset_name = config.dataset_name
        self.transform = transforms.Compose([
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Filter the dataset for the specified class
        self.data = []
        count = 0
        for i in range(len(dataset)):
            if count >= n_samples:
                break
            if dataset[i]['label'] == class_label:
                self.data.append(dataset[i])
                count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_key = 'img' if 'cifar10' == self.dataset_name else 'image'
        image = self.transform(item[image_key].convert("RGB"))
        label = item['label']
        return image, label
    
    def shuffle(self):
        random.shuffle(self.data)
        
# TODO: delete this later        
'''
def compute_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
'''

def create_conditional_subdatasets(dataset, config, class_labels, samples_per_class):
    subdatasets = []
    for class_label in class_labels:
        subdataset = ConditionalSubdataset(dataset, config, class_label, samples_per_class)
        subdatasets.append(subdataset)
    return subdatasets



# Each Subdataset contain equally distributed data with designated class_labels (2 class case with [0, 1], for example, half zeros and ones should be in the first subset, and the residue zeros and ones should be in the second subset)
# should be work even when n_class and n_subset are different (ex. n_class=4, n_subset=2)
# raise error when n_samples_per_subset * n_subsets > n_samples_per_class * n_class
class UnconditionalSubdataset(Dataset):
    def __init__(self, dataset, config, n_samples_per_subset, index_per_class, class_labels):
        self.dataset_name = config.dataset_name
        self.transform = transforms.Compose([
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.data = []
        n_samples_per_class = n_samples_per_subset // len(class_labels)

        for n, class_label in enumerate(class_labels):
            count = 0
            for i in range(index_per_class[n], len(dataset)):
                if count >= n_samples_per_class:
                    index_per_class[n] = i
                    break
                if dataset[i]['label'] == class_label:
                    self.data.append(dataset[i])
                    count += 1

    def __len__(self):
        return len(self.data);

    def __getitem__(self, idx):
        item = self.data[idx]
        image_key = 'img' if 'cifar10' == self.dataset_name else 'image'
        image = self.transform(item[image_key].convert("RGB"))
        label = item['label']
        return image, label
    
    def shuffle(self):
        random.shuffle(self.data)

def create_unconditional_subdatasets(dataset, config, class_labels, n_subsets, n_samples_per_subset):
    subdatasets = []
    index_per_class = torch.zeros(len(class_labels), dtype=torch.int32)
    for k in range(n_subsets):
        subdataset = UnconditionalSubdataset(dataset, config, n_samples_per_subset, index_per_class, class_labels)
        print(index_per_class, len(subdataset))
        subdatasets.append(subdataset)
    return subdatasets


# return mini-batch with equally distributed data from every subdatasets.
# conditional 2 class case with [0, 1], for example, every mini-batch contains half 0 and half 1 retrieved from each subdatasets.
# same process will be applied in the unconditional setup.
class BalancedDataloader(IterableDataset):
    def __init__(self, subdatasets, batch_size):
        self.subdatasets = subdatasets
        self.batch_size = batch_size
        self.k = len(subdatasets)
        self.iterators = [iter(dataset) for dataset in subdatasets]
        self.total_length = sum(len(d) for d in subdatasets) // self.batch_size
        self.dataset_exhausted = [False] * self.k  # Track whether each dataset is exhausted

    def __iter__(self):
        for subdataset in self.subdatasets:
            subdataset.shuffle()
        
        self.iterators = [iter(dataset) for dataset in self.subdatasets]  # Reset iterators
        self.dataset_exhausted = [False] * self.k  # Reset exhausted flags
        return self

    def __next__(self):
        if all(self.dataset_exhausted):  # Check if all datasets are exhausted
            raise StopIteration
        
        inputs, labels, subset_labels = [], [], []
        for _ in range(self.batch_size // self.k):
            for dataset_index, it in enumerate(self.iterators):
                if self.dataset_exhausted[dataset_index]:
                    continue  # Skip dataset if it's exhausted
                
                try:
                    image, label = next(it)                    
                except StopIteration:
                    self.dataset_exhausted[dataset_index] = True  # Mark dataset as exhausted
                    continue

                inputs.append(image)
                labels.append(label)
                subset_labels.append(dataset_index)  # Track subdataset index
                
        if not inputs:  # Check if no data was added (all datasets exhausted)
            raise StopIteration

        # Shuffle the batch to mix data points from different datasets
        combined = list(zip(inputs, labels, subset_labels))
        random.shuffle(combined)
        inputs, labels, subset_labels = zip(*combined)

        # Convert lists to tensors
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)
        subset_labels = torch.tensor(subset_labels)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        labels = labels.to(device)
        subset_labels = subset_labels.to(device)

        return {"input": inputs, "labels": labels, "subset_labels": subset_labels}
    
    def __len__(self):
        return self.total_length






