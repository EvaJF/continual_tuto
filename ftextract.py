import torch
import torch.nn as nn
import sys
from utils_tuto.parser import parse_args
from utils_tuto.encoder import get_encoder, encode_features
from utils_tuto.dataset import get_dataset
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
    print('\nPreparing dataset...')
    dataset_train, dataset_val, dataset_test = get_dataset(args.dataset, data_transforms)

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
        my_net, loader_train, save_dir=f"data/features/{args.dataset}/{args.archi}/{args.pretrain}/train", device=device
    )
    print("Completed feature extraction for Train images\n")
    encode_features(
        my_net, loader_test, save_dir=f"data/features/{args.dataset}/{args.archi}/{args.pretrain}/test", device=device
    )
    print("Completed feature extraction for Test images\n")
    #if valid : 
    encode_features(
        my_net, loader_val, save_dir=f"data/features/{args.dataset}/{args.archi}/{args.pretrain}/valid", device=device
    )
    print("Completed feature extraction for Valid images\n")

            

if __name__ == "__main__":
    main()
    sys.exit()