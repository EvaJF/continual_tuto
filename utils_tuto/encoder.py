import os
import torch
from tqdm import tqdm
import timm
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


# minimal working example with a small CNN
class myCNN(nn.Module):
    def __init__(self, nb_tot_cl, size_conv_1, size_conv_2, size_fc):
        super(myCNN, self).__init__()
        self.nb_tot_cl = nb_tot_cl
        self.size_conv_2 = size_conv_2
        self.conv_1 = nn.Conv2d(
            1, size_conv_1, 5, 1
        )  # in_channels, out_channels, kernel_size, stride
        self.conv_2 = nn.Conv2d(size_conv_1, size_conv_2, 5, 1)
        # self.drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(4 * 4 * size_conv_2, size_fc)
        self.fc = nn.Linear(size_fc, nb_tot_cl)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        # x = F.relu(self.drop(self.conv_2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * self.size_conv_2)
        x = F.relu(self.fc_1(x))
        x = self.fc(x)
        return x

    def update_fc(self, nb_incr_cl):
        """
        Replaces model.fc with a new fc layer with an additional nb_incr_cl classes.
        Keeps original weights for the previous classes.
        """

        old_fc = self.fc
        nb_old_cl = old_fc.out_features
        in_features = old_fc.in_features
        new_fc = nn.Linear(in_features, nb_old_cl + nb_incr_cl)

        # Copy existing weights and biases
        with torch.no_grad():
            new_fc.weight[:nb_old_cl] = old_fc.weight
            new_fc.bias[:nb_old_cl] = old_fc.bias

        self.fc = new_fc


# Change the format of the forward for easy KD loss on features
class myCNN_features(nn.Module):
    def __init__(self, nb_tot_cl, size_conv_1, size_conv_2, size_fc):
        super(myCNN_features, self).__init__()
        self.nb_tot_cl = nb_tot_cl
        self.size_conv_2 = size_conv_2
        self.conv_1 = nn.Conv2d(
            1, size_conv_1, 5, 1
        )  # in_channels, out_channels, kernel_size, stride
        self.conv_2 = nn.Conv2d(size_conv_1, size_conv_2, 5, 1)
        # self.drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(4 * 4 * size_conv_2, size_fc)
        self.fc = nn.Linear(size_fc, nb_tot_cl)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        # x = F.relu(self.drop(self.conv_2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * self.size_conv_2)
        feat = F.relu(self.fc_1(x))
        out = self.fc(feat)
        return {"features" : feat, "out" : out}

    def update_fc(self, nb_incr_cl):
        """
        Replaces model.fc with a new fc layer with an additional nb_incr_cl classes.
        Keeps original weights for the previous classes.
        """

        old_fc = self.fc
        nb_old_cl = old_fc.out_features
        in_features = old_fc.in_features
        new_fc = nn.Linear(in_features, nb_old_cl + nb_incr_cl)

        # Copy existing weights and biases
        with torch.no_grad():
            new_fc.weight[:nb_old_cl] = old_fc.weight
            new_fc.bias[:nb_old_cl] = old_fc.bias

        self.fc = new_fc


def get_encoder(archi, pretrain):

    if archi == "resnet18":  # use torch
        if pretrain == "in1k":  # IMAGENET1K_V1
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            trf = weights.transforms()
        elif pretrain == "none":  # minimalistic transform
            # by default, use imagenet stats for normalizing input images
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            trf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize([224, 224]),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            raise NotImplementedError(
                "Seems like the pretraining you are looking for is missing"
            )

    elif archi == "vits":  # use timm for ViT
        if pretrain == "in1k":  # ILSVRC
            model_name = "vit_small_patch16_224.augreg_in1k"
        elif pretrain == "in21k":  # ImageNet-21k
            model_name = "vit_base_patch16_224.augreg_in21k"
        elif pretrain == "lvd142m":  # LVD 142 million
            model_name = "vit_small_patch14_dinov2.lvd142m"
        else:
            raise NotImplementedError(
                "Seems like the pretraining you are looking for is missing"
            )
        model = timm.create_model(model_name, pretrained=True)

        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        trf = timm.data.create_transform(**data_config, is_training=False)

    else:
        return NotImplementedError(
            "Seems like the architecture you are looking for is missing"
        )

    return model, trf


def encode_features(encoder, loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    encoder = encoder.to(device)
    encoder.eval()
    last_class = -1

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            features = encoder(inputs)
            labels = labels.tolist()
            features = features.tolist()
            for i in range(len(labels)):
                curr_class = labels[i]
                if curr_class != last_class:
                    last_class = curr_class
                with open(os.path.join(save_dir, str(curr_class)), "a") as features_out:
                    features_out.write(
                        str(" ".join([str(e) for e in list(features[i])])) + "\n"
                    )

    return None


def compute_num_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal number of parameters: {tot}")
    print(
        f"Nb of trainable parameters  : {trainable} ({np.round(100*trainable/tot, 2)}%)"
    )
    return None
