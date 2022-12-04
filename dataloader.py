import os
from torch.utils.data import Dataset, DataLoader, random_split
import natsort
from PIL import Image
from torchvision import transforms, datasets, models
import math
import warnings
from torch import default_generator, randperm
from torch.utils.data.dataset import Subset
from torch._utils import _accumulate
from typing import List

# Custom class to read in data
class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

    def get_file_name(self):
        return self.main_dir


def random_split_dh(dataset, lengths, generator=default_generator):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths) 
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def load_data(args):
    '''
    Function to load in our training data
    Parameters
    ----------
    args: parameters passed in by the user

    Returns
    -------
    training dataset: owners images (clean images) and finders images (noisy images) of pets for training 
    testing dataset: owners images (clean images) and finders images (noisy images) of pets for testing
    '''
    print("reading in data")
    
    # check to see how many images without backgrounds we have. Should be 49,988
    train_dir_nobg = "./train_no_bg"
    count = 0
    for file_name in os.listdir(train_dir_nobg):
        count = count + 1
    print(f"{count} number of images in folder")


    # get data into right format
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # read in data
    dataset = CustomDataSet(train_dir_nobg, transform=data_transform)
    print(f"full dataset size: {dataset.__len__()}")
    print(f"image shape: {dataset[0].shape}")

    # Set up training and testing sets
    data_size = dataset.__len__()

    # Random split dataset
    train_dataset, validation_dataset = random_split_dh(dataset, [round(0.2 * data_size), round(0.03 * data_size)])  # n=10,000 to train, n = 1500 to test

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)

    print(f"length of training dataloader {len(train_loader)}")
    print(f"length of testing dataloader {len(test_loader)}")

    return train_loader, test_loader