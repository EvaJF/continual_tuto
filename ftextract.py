import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import sys
from sklearn.model_selection import train_test_split
from utils_tuto.parser import parse_args
from utils_tuto.encoder import get_encoder, encode_features
from utils_tuto.utils import get_device, seed_all

def main():
    args = parse_args()
    seed_all()
    
    # if available, use GPU / MPS, else CPU
    print(f"Torch default device: {torch.get_default_device()}")
    device = get_device()
    print(f"Selected device {device}")

    # get a (pre-trained) encoder
    print(f"\nLoading pre-trained model with architecture {args.archi}, pretrained on dataset {args.pretrain}")
    my_net, data_transforms = get_encoder(args.archi, args.pretrain) 

    # keep only the encoder / replace the classification head by an Identity layer
    my_net.fc = nn.Identity()
    print(my_net)
    
    # dataset
    dataset = args.dataset
    print('\nPreparing dataset...')

    if dataset == "food101" : 
        dataset_train = datasets.Food101(
            root="data",
            split = 'train',
            download=True,
            transform=data_transforms
        )

        dataset_val = datasets.Food101(
            root="data",
            split='train',
            download=False,
            transform=data_transforms
        )

        dataset_test = datasets.Food101(
            root="data",
            split='test',
            download=True,
            transform=data_transforms
        )    
        #print(dataset_train.__dir__())

        valid = False
        # custom validation dataset
        if valid : 
            samples_train, samples_val = train_test_split(dataset_train._image_files, test_size=0.2)
            dataset_train._image_files = samples_train
            dataset_val._image_files = samples_val     

    elif dataset == "flowers102": 
        dataset_train = datasets.Flowers102(
            root="data",
            split='train',
            download=True,
            transform=data_transforms
        )

        dataset_val = datasets.Flowers102(
            root="data",
            split='val',
            download=True,
            transform=data_transforms
        )

        dataset_test = datasets.Flowers102(
            root="data",
            split='test',
            download=True,
            transform=data_transforms
        )   
        valid = True 
    else : 
        raise NotImplementedError("This dataset name doesn't seem to be supported. Implement a custom Dataset.")
        

    # datasets and loaders from train / val / test image lists
    print("Nb of train images : %i" % len(dataset_train))
    print("Nb of val images : %i" % len(dataset_val))
    print("Nb of test images : %i" % len(dataset_test))   

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=4
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=32, shuffle=False, num_workers=4
    )

    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=4
    )

    # Feature extraction
    print("\nFeature extraction...")
    encode_features(
        my_net, loader_train, save_dir=f"data/features/{dataset}/train", device=device
    )
    print("Completed feature extraction for Train images\n")
    encode_features(
        my_net, loader_test, save_dir=f"data/features/{dataset}/test", device=device
    )
    print("Completed feature extraction for Test images\n")
    if valid : 
            encode_features(
                my_net, loader_val, save_dir=f"data/features/{dataset}/valid", device=device
            )
            print("Completed feature extraction for Valid images\n")

            

if __name__ == "__main__":
    main()
    sys.exit()