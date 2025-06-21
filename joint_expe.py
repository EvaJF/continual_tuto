#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Joint training using MNIST dataset
#
# based on  https://github.com/hleborgne/TDDL/blob/master/2_usual_DNN/mnist_pytorch_exercise.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from utils_tuto.utils import get_device, seed_all
from utils_tuto.encoder import compute_num_params, myCNN
from time import time
import copy
import pandas as pd

# we use GPU or MPS if available, otherwise CPU
device = get_device()
print(f"Running on device: {device}")

# mean and stdev of pixel values
trf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# dataset -> download images
train_set = datasets.MNIST("./data", train=True, transform=trf, download=True)
test_set = datasets.MNIST("./data", train=False, transform=trf, download=True)

# split train set in train/val
seed_all(42)
val_set = datasets.MNIST("./data", train=True, transform=trf, download=False)
index_list = [k for k in range(len(train_set))]
train_index, val_index = train_test_split(index_list, test_size=0.1)

train_set.data = train_set.data[train_index]
train_set.targets = train_set.targets[train_index]
val_set.data = val_set.data[val_index]
val_set.targets = val_set.targets[val_index]

print("Nb of train images : %i" % len(train_set))
print("Nb of val images : %i" % len(val_set))
print("Nb of test images : %i" % len(test_set))

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

print("training batch number: {}".format(len(train_loader)))
print("testing batch number: {}".format(len(test_loader)))


# define model : minimal working example with a small CNN
nb_tot_cl = 10
size_conv_1 = 8
size_conv_2 = 16
size_fc = 256
model = myCNN(nb_tot_cl, size_conv_1, size_conv_2, size_fc)
print(model)
compute_num_params(model)

model.to(device)  # puts model on GPU / CPU

# optimization
lr = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # try lr=0.01
loss_fn = nn.CrossEntropyLoss()

# main loop (train/val)
EPOCHS = 10
start = time()
best_acc, best_model_state = 0.0, None
acc_list = []

print("\nTraining...")
for epoch in range(EPOCHS):
    # training
    model.train()
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "epoch {:2d} batch {:3d} [{:5d}/{:5d}] training loss: {:0.4f}".format(
                    epoch,
                    batch_idx,
                    batch_idx * len(x),
                    len(train_loader.dataset),
                    loss.item(),
                )
            )
    # eval on val set
    model.eval()
    correct = 0
    with torch.no_grad():
        confusion = torch.zeros(nb_tot_cl, nb_tot_cl)
        for batch_idx, (x, target) in enumerate(val_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            prediction = out.argmax(
                dim=1, keepdim=True
            )  # index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            for i, j in zip(prediction, target):
                confusion[i.to("cpu"), j.to("cpu")] += 1
    acc = 100.0 * correct / len(val_loader.dataset)
    acc_list.append(acc)
    # save the curent best model checkpoint
    if acc > best_acc:
        best_acc = acc
        best_model_state = copy.deepcopy(model.state_dict())
    # display
    print("Val Acc: {:.2f}% ({}/{})".format(acc, correct, len(val_loader.dataset)))
    torch.set_printoptions(sci_mode=False)
    print(f"Confusion matrix at epoch {epoch}")
    print(confusion.int().numpy())

print(f"\nVal Acc history : {acc_list}")
print("Best val acc {:.2f}".format(best_acc))

# save best checkpoint
os.makedirs("ckp", exist_ok=True)
ckp_path = "ckp/mnist_joint_best.pt"
print(
    "\nSaving best model checkpoint under {}".format(
        os.path.join(os.getcwd(), ckp_path)
    )
)
torch.save(best_model_state, ckp_path)

# reload best ckp
# model.load_state_dict(best_model_state)
model.load_state_dict(torch.load(ckp_path, weights_only=True))

# final eval on test set
print("\nTesting the best model...")
model.eval()
correct = 0
with torch.no_grad():
    confusion = torch.zeros(nb_tot_cl, nb_tot_cl)
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        out = model(x)
        loss = loss_fn(out, target)
        prediction = out.argmax(dim=1, keepdim=True)  # index of the max log-probability
        correct += prediction.eq(target.view_as(prediction)).sum().item()
        for i, j in zip(prediction, target):
            confusion[i.to("cpu"), j.to("cpu")] += 1
test_acc = 100.0 * correct / len(test_loader.dataset)
print("Test Acc: {:.2f}% ({}/{})".format(test_acc, correct, len(test_loader.dataset)))
torch.set_printoptions(sci_mode=False)
print(f"Confusion matrix on test set")
print(confusion.int().numpy())

print(f"\n=== Report - Joint training ===\n")
print("Hyperparameters : epochs {}, lr {}, archi CNN {}/{}/{}".format(EPOCHS, lr, size_conv_1, size_conv_2, size_fc))
print(f"\n>> Test accuracy : {{:.2f}} <<".format(test_acc))

data = {
    "method" : ["joint"],
    "nb_init_cl" : [nb_tot_cl],
    "nb_incr_cl" : [0],
    "nb_tot_cl" : [nb_tot_cl],
    "memory" : [0], 
    "CE_loss" : ["classic"],
    "KD_loss" : ["None"],
    "lr" : [lr], 
    "epochs" : [EPOCHS],
    "last_acc" : [test_acc],
    "avg_incr_acc" : [0.0],
    "avg_f" : [0.0]
}
df = pd.DataFrame.from_dict(data)
print(df)
os.makedirs('logs', exist_ok=True)
df.to_csv(os.path.join('logs', 'results.csv'))

elapsed = (time() - start) / 60  # elapsed time in minutes
print(f"\nCompleted expe in {{:.2f}} min".format(elapsed))
