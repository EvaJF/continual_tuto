import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import shutil
from PIL import Image
from typing import Union, Optional, Callable
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision import datasets


class incrMNIST(MNIST):
    def __init__(
        self,
        root: Union[str, Path],
        class_range: range,
        replay: int = 0,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        remap_labels: bool = False,  # Optional: remap labels to 0, 1, ..., N-1
    ) -> None:
        if not isinstance(class_range, range):
            raise TypeError(
                f"`class_range` must be of type `range`, got {type(class_range)}"
            )

        self.class_range = class_range  # class range of samples
        self.replay = replay  # number of samples in the memory buffer
        self.remap_labels = remap_labels
        self._class_list = list(class_range)
        self._class_set = set(self._class_list)
        self.folder_name = "MNIST"  # do not download again MNIST dataset
        self.replay_targets = None
        self.replay_data = None

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # select the training examples from the current step
        # if using a replay strategy, add a subset of past samples
        self._filter_by_class_range()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.folder_name, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.folder_name, "processed")

    def _filter_by_class_range(self) -> None:
        """
        Filters self.data and self.targets to only include samples
        whose labels are in self.class_range.
        Optionally adds 'replay' samples from earlier classes.

        Assumes incremental training follows label order (i.e., class 0, 1, 2, ...).

        NB : current replay strategy is random
        """
        targets = self.targets
        class_tensor = torch.tensor(
            self._class_list, dtype=targets.dtype, device=targets.device
        )
        mask = torch.isin(targets, class_tensor)
        data = self.data[mask]
        targets = self.targets[mask]

        # quick and dirty ! explain why
        if (self.replay > 0) and (self.train is True) and (min(self._class_list) > 0):
            # select samples with label < min of current class list
            buffer_classes = list(range(min(self._class_list)))
            print("Buffer classes:", buffer_classes)

            # mask for all original targets (not filtered ones)
            buffer_tensor = torch.tensor(
                buffer_classes, dtype=self.targets.dtype, device=self.targets.device
            )
            buffer_mask = torch.isin(self.targets, buffer_tensor)

            # indices of samples in buffer classes
            buffer_indices = torch.nonzero(buffer_mask, as_tuple=False).squeeze()

            if len(buffer_indices) > 0:
                num_samples = min(self.replay, len(buffer_indices))
                sampled_indices = buffer_indices[
                    torch.randperm(len(buffer_indices))[:num_samples]
                ]

                buffer_data = self.data[sampled_indices]
                buffer_targets = self.targets[sampled_indices]

                print(
                    "Sanity check - buffer class range:",
                    buffer_targets.min(),
                    buffer_targets.max(),
                )
                print("Buffer sample shapes:", buffer_data.shape, buffer_targets.shape)

                data = torch.cat([data, buffer_data])
                targets = torch.cat([targets, buffer_targets])
            else:
                print("No buffer samples available for replay.")

        self.data = data
        self.targets = targets

        if self.remap_labels:
            # Map class_range to contiguous labels starting from 0
            label_mapping = {
                original: idx for idx, original in enumerate(self._class_list)
            }
            self.targets = torch.tensor(
                [label_mapping[int(label)] for label in self.targets], dtype=torch.long
            )


def get_MNIST_loaders(train_range, test_range, batch_size=64, root="./data"):

    # mean and stdev of pixel values
    trf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # custom dataset : select a range of classes
    train_set = incrMNIST(
        root=root, class_range=train_range, train=True, transform=trf, download=False
    )  # we have already downloaded the data

    val_set = incrMNIST(
        root=root, class_range=train_range, train=True, transform=trf, download=False
    )

    test_set = incrMNIST(
        root=root, class_range=test_range, train=False, transform=trf, download=False
    )

    # split train set in train/val sets
    index_list = [k for k in range(len(train_set))]
    train_index, val_index = train_test_split(index_list, test_size=0.1)

    train_set.data = train_set.data[train_index]
    train_set.targets = train_set.targets[train_index]
    val_set.data = val_set.data[val_index]
    val_set.targets = val_set.targets[val_index]

    print("Nb of train samples: %i" % len(train_set))
    print("Nb of val samples  : %i" % len(val_set))
    print("Nb of test samples : %i" % len(test_set))

    # define data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("Nb training batches: {}".format(len(train_loader)))
    print("Nb testing batches : {}".format(len(test_loader)))

    return train_loader, val_loader, test_loader


def class_count_dict(targets: torch.Tensor) -> dict:
    """
    Returns a dictionary {label: count} for the given targets tensor.
    """
    cc_dict = {k: 0 for k in range(torch.max(targets).item())}
    labels, counts = torch.unique(targets, return_counts=True)
    cc_dict.update({label.item(): count.item() for label, count in zip(labels, counts)})
    return cc_dict


class Memory(Dataset):
    def __init__(
        self,
        max_size: int,
        cumul: bool = False,
        nb_new: int = 10,
        strategy: str = "random",
        steps: int = 0,
    ):
        """
        Memory buffer

        max_size (int) : the maximum number of samples that the memory should contain
        cumul (bool) : whether the memory buffer is fed in a cumulative manner, or has always the max_size number of samples
        nb_new (int) : the number of new samples to select for feeding the memory buffer
        strategy (str), default "random" : how to select the samples for feeding the memory buffer
        steps (int), default 0 : the number of times the buffer has been updated

        """
        assert max_size > 0, "max_size must be a positive integer."
        assert nb_new >= 0, "nb_new must be non-negative."
        assert strategy == "random", "Only 'random' strategy is implemented."

        self.max_size = max_size
        self.cumul = cumul
        self.nb_new = nb_new
        self.strategy = strategy
        self.steps = steps

        self.data = torch.empty(0)
        self.targets = torch.empty(0, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def feed_memory(self, D: Dataset) -> None:
        assert hasattr(D, "data") and hasattr(
            D, "targets"
        ), "Input dataset must have 'data' and 'targets'."
        D_data = D.data
        D_targets = D.targets
        total_in_D = len(D_data)

        print(f"\n[Memory] Step {self.steps + 1}")
        if len(self.targets) > 0:
            print("Before update - class distribution:", class_count_dict(self.targets))

        self.steps += 1

        if self.strategy != "random":
            raise NotImplementedError("Only 'random' strategy is implemented.")

        # Step 1: filter the incoming dataset based on the memory state
        # to only add samples from new classes into memory
        if len(self.targets) > 0:
            threshold = self.targets.max().item()
            mask = D_targets > threshold
            D_data = D_data[mask]
            D_targets = D_targets[mask]

        total_in_D = len(D_data)
        if total_in_D == 0:
            return  # no eligible new samples, nothing to add

        # Step 2 : update the memory buffer
        if (
            self.cumul
        ):  # add a fixed number of new samples per update in a cumulative manner
            if len(self.data) + self.nb_new > self.max_size:
                print("Reached max size, removing some samples")
                # Keep only (max_size - nb_new) from current memory
                keep_n = self.max_size - self.nb_new
                if keep_n <= 0:  # saturated memory
                    raise ValueError("nb_new exceeds max_size in cumulative mode.")
                if keep_n > 0:
                    perm = torch.randperm(len(self.data))[:keep_n]
                    self.data = self.data[perm]
                    self.targets = self.targets[perm]

            # Select nb_new samples from D
            sample_n = min(self.nb_new, total_in_D)
            new_indices = torch.randperm(total_in_D)[:sample_n]
            new_data = D_data[new_indices]
            new_targets = D_targets[new_indices]

            # Append to memory
            self.data = torch.cat([self.data, new_data])
            self.targets = torch.cat([self.targets, new_targets])

        else:  # maintain max_size samples in memory
            if len(self.data) == 0:
                # First fill
                sample_n = min(self.max_size, total_in_D)
                new_indices = torch.randperm(total_in_D)[:sample_n]
                self.data = D_data[new_indices]
                self.targets = D_targets[new_indices]
            else:
                # Remove and replace equal number of samples
                replace_n = self.max_size // (self.steps)
                replace_n = min(replace_n, min(len(self.data), total_in_D))

                if replace_n > 0:
                    keep_indices = torch.randperm(len(self.data))[
                        : len(self.data) - replace_n
                    ]
                    self.data = self.data[keep_indices]
                    self.targets = self.targets[keep_indices]

                    new_indices = torch.randperm(total_in_D)[:replace_n]
                    new_data = D_data[new_indices]
                    new_targets = D_targets[new_indices]

                    self.data = torch.cat([self.data, new_data])
                    self.targets = torch.cat([self.targets, new_targets])

        if len(self.targets) > 0:
            print("After update  - class distribution:", class_count_dict(self.targets))


def get_MNIST_loaders_with_memory(
    train_range, test_range, memory, batch_size=64, root="./data"
):
    # mean and stdev of pixel values
    trf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # custom dataset : select a range of classes
    train_set = incrMNIST(
        root=root, class_range=train_range, train=True, transform=trf, download=False
    )

    train_subset = incrMNIST(
        root=root, class_range=train_range, train=True, transform=trf, download=False
    )

    val_subset = incrMNIST(
        root=root, class_range=train_range, train=True, transform=trf, download=False
    )

    test_set = incrMNIST(
        root=root, class_range=test_range, train=False, transform=trf, download=False
    )

    # add memory buffer to train_set
    print("Memory size        : %i" % len(memory))
    print("Nb of train samples: %i" % len(train_set))
    if len(memory) > 0:
        train_set.data = torch.cat([train_set.data, memory.data])
        train_set.targets = torch.cat([train_set.targets, memory.targets])
    print("Total train samples: %i" % len(train_set))

    # split updated train_set into train and val sets
    index_list = list(range(len(train_set)))
    train_index, val_index = train_test_split(index_list, test_size=0.1)

    train_subset.data = train_set.data[train_index]
    train_subset.targets = train_set.targets[train_index]
    val_subset.data = train_set.data[val_index]
    val_subset.targets = train_set.targets[val_index]

    print("Nb of train samples: %i" % len(train_subset))
    print("Nb of val samples  : %i" % len(val_subset))
    print("Nb of test samples : %i" % len(test_set))

    # define data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("Nb training batches: {}".format(len(train_loader)))
    print("Nb testing batches : {}".format(len(test_loader)))

    return train_loader, val_loader, test_loader


def get_dataset(dataset, data_transforms, val_size=0.1):
    if dataset == "mnist":
        dataset_train = datasets.MNIST(
            "./data", train=True, transform=data_transforms, download=True
        )
        dataset_test = datasets.MNIST(
            "./data", train=False, transform=data_transforms, download=True
        )
        # custom validation set
        dataset_val = datasets.MNIST(
            "./data", train=True, transform=data_transforms, download=False
        )
        index_list = [k for k in range(len(dataset_train))]
        train_index, val_index = train_test_split(index_list, test_size=val_size)

        dataset_train.data = dataset_train.data[train_index]
        dataset_train.targets = dataset_train.targets[train_index]
        dataset_val.data = dataset_val.data[val_index]
        dataset_val.targets = dataset_val.targets[val_index]

    elif dataset == "food-101":
        dataset_train = datasets.Food101(
            root="data", split="train", download=True, transform=data_transforms
        )

        dataset_val = datasets.Food101(
            root="data", split="train", download=False, transform=data_transforms
        )

        dataset_test = datasets.Food101(
            root="data", split="test", download=True, transform=data_transforms
        )
        # print(dataset_train.__dir__())

        # custom validation dataset
        samples_train, samples_val = train_test_split(
            dataset_train._image_files, test_size=val_size
        )
        dataset_train._image_files = samples_train
        dataset_val._image_files = samples_val

    elif dataset == "flowers-102":
        dataset_train = datasets.Flowers102(
            root="data", split="train", download=True, transform=data_transforms
        )

        dataset_val = datasets.Flowers102(
            root="data", split="val", download=True, transform=data_transforms
        )

        dataset_test = datasets.Flowers102(
            root="data", split="test", download=True, transform=data_transforms
        )

    else:
        raise NotImplementedError(
            "This dataset name doesn't seem to be supported. Implement a custom Dataset."
        )
    return dataset_train, dataset_val, dataset_test


class ImageDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        """
        Inputs
        ------
        samples :
        lables :
        transform :

        """
        self.labels = labels
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Select sample
        img = self.samples[index]
        # Load data and get label
        X = Image.open(img)
        X = self.transform(X)
        y = self.labels[img]

        return X, y


class FeaturesDataset(Dataset):
    def __init__(self, path, range_classes=None):
        self.X, self.y = [], []
        dir_list = os.listdir(path)
        if range_classes:
            dir_list = [f"{d}_untied" for d in range_classes]
        dir_list = np.sort(dir_list)
        for root_path in dir_list:
            if "_" in root_path:
                # print(root_path, os.listdir(os.path.join(path, root_path)))
                for file in os.listdir(os.path.join(path, root_path)):
                    self.X.append(os.path.join(path, root_path, file))
                    self.y.append(int(root_path.split("_")[0]))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).astype(np.float32)
        return image, label


def read_features_as_df(path, n):
    """
    One row = one example
    One col = one feature
    """
    L = []
    for c in range(n):
        with open(os.path.join(path, str(c)), "r") as f:
            lines = f.readlines()
            # print(len(lines))
        lines = [[float(x) for x in string.split(" ")] + [c] for string in lines]
        # print(len(lines))
        features = np.array(lines)
        # print(features.shape)
        L.append(features)
    print("individual features.shape", features.shape)
    F = np.vstack(L)
    df = pd.DataFrame(F)
    return df


def read_features_as_dic(path, n):
    """
    data handling : open saved features
    --> dic of arrays of size n_samples * n_hidden
    """
    features_dic = {}
    for c in range(n):
        with open(os.path.join(path, str(c)), "r") as f:
            lines = f.readlines()
            lines = [[float(x) for x in string.split(" ")] for string in lines]
        features = np.array(lines)
        features_dic[c] = features
    return features_dic


def read_features_as_array(path, n):
    features_dic = read_features_as_dic(path, n)
    X_features = np.concatenate([features_dic[key] for key in features_dic.keys()])
    print("X_features.shape", X_features.shape)
    true_labels = np.concatenate(
        [[key for i in range(len(features_dic[key]))] for key in features_dic.keys()]
    )
    print("len(true_labels)", len(true_labels))
    # print(true_labels[::50])
    return X_features, true_labels


def untie_features(n, train_features_path, val_features_path):
    for dir in [train_features_path, val_features_path]:
        file_path = os.path.join(dir, str(n))
        if os.path.exists(file_path):
            try:
                shutil.rmtree(file_path + "_untied")
            except:
                pass
            os.makedirs(file_path + "_untied", exist_ok=True)
            c = 0
            with open(file_path, "r") as f:
                for line in f:
                    with open(os.path.join(file_path + "_untied", str(c)), "w") as f2:
                        f2.write(line)
                    c += 1
    return None


def clean_features(n, train_features_path, val_features_path):
    for dir in [train_features_path, val_features_path]:
        file_path = os.path.join(dir, str(n))
        if os.path.exists(file_path):
            try:
                shutil.rmtree(file_path + "_untied")
                # print("deleted :", sscratch_file_path+'_untied')
            except:
                print("Could not clean", file_path)
                pass
