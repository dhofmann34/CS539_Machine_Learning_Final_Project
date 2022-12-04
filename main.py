# Final project for CS 539 
import argparse
from pydoc import importfile
from random import random
import random
import numpy as np
import torch

from dataloader import load_data
from model import train

# the following code was used to train the autoencoder. The data preparation and evaluation can be found in the jupyter notebook.

if __name__ == "__main__":

    # parsing and running configuration
    parser = argparse.ArgumentParser(description="Pet Finder")
    parser.add_argument('--epochs', type=int, default=200,help='number of epochs to train (default: 20)')
    parser.add_argument('--seed', default=1, type=int,help='seed for initializing training')  # set seed for training here

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)  
    print(args)

    if torch.cuda.is_available():
        print(f"Running on {torch.cuda.get_device_name(0)}")
    else:
        print(f"Running on CPU")

    # load in dataset
    train_loader, test_loader = load_data(args)

    # train autoencoder
    train(args, train_loader)