#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Fine-tuning with replay using MNIST dataset

from time import time
import copy
import os
import numpy as np
import torch
import torch.nn as nn
from utils_tuto.utils import get_device, seed_all
from utils_tuto.encoder import compute_num_params, myCNN
from utils_tuto.dataset import get_MNIST_loaders_with_memory, Memory, class_count_dict
from utils_tuto.perf import compute_forgetting
from utils_tuto.loss import BalancedCrossEntropy
import pandas as pd

seed_all(42)

# we use GPU or MPS if available, otherwise CPU
device = get_device()
print(f"Running on device: {device}")

# define incremental scenario
nb_init_cl = 2  # nb of initial classes
nb_tot_cl = 10  # total nb of classes
nb_incr_cl = 2  # nb of classes per update
nb_steps = (nb_tot_cl - nb_init_cl) // nb_incr_cl  # number of incremental steps

# replay strategy
max_size = 200  # try 800, 2000
cumul = False 
CE_type = "classic" # "weighted", "balanced" 
assert CE_type in ["weighted", "classic", "balanced"]

# define model architecture
size_conv_1 = 8
size_conv_2 = 16
size_fc = 256

# training hyperparameters
EPOCHS = 4  # 2
lr = 0.01
momentum = 0.9

# ckp path
ckp_root = os.path.join(os.getcwd(), "ckp")

start = time()

acc_mat = np.zeros((nb_steps + 1, nb_steps + 1))
test_acc_list = []
nb_test_samples = []

# incremental training loop
for step in range(nb_steps + 1):
    nb_curr_cl = nb_init_cl + nb_incr_cl * step
    print(f"\n== Step {step} ==")

    if step == 0:
        train_cl = range(nb_init_cl)
        test_cl = range(nb_init_cl)
        # initialize model
        model = myCNN(nb_init_cl, size_conv_1, size_conv_2, size_fc)
        # initialize memory buffer
        memory = Memory(max_size=max_size)
    else:
        train_cl = range(nb_init_cl + nb_incr_cl * (step - 1), nb_curr_cl)
        test_cl = range(nb_curr_cl)
        # new fc layer : add weights for the new classes
        # keep the weights of the clasification layer for the previous classes
        model.update_fc(nb_incr_cl)
        # update the memory buffer with current training samples
        print("Memory size before update {}".format(len(memory)))
        memory.feed_memory(train_loader.dataset)
        print("Memory size after update  {}".format(len(memory)))

    print(model)
    compute_num_params(model)
    model.to(device)  # puts model on GPU / CPU

    # dataloader
    print(f"Training on classes {train_cl}")
    print(f"Testing on classes {test_cl}")
    train_loader, val_loader, test_loader = get_MNIST_loaders_with_memory(
        train_range=train_cl, test_range=test_cl, memory=memory
    )
    train_count_dict = class_count_dict(train_loader.dataset.targets)
    val_count_dict = class_count_dict(val_loader.dataset.targets)
    test_count_dict = class_count_dict(test_loader.dataset.targets)
    print(f"Train :{train_count_dict}")
    print(f"Val   :{val_count_dict}")
    print(f"Test  :{test_count_dict}")

    # optimization
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if CE_type == "classic" : 
        loss_fn = nn.CrossEntropyLoss()
    elif CE_type == "weighted" : 
        CE_weights = [ 1-train_count_dict[k]/len(train_loader.dataset) for k in range(nb_curr_cl)]
        print(f"Class weights {CE_weights}")
        CE_weights = torch.tensor(CE_weights, device=device)
        loss_fn = nn.CrossEntropyLoss(weight = CE_weights)
    elif CE_type == "balanced" : 
        freq = torch.tensor([train_count_dict[k] for k in range(nb_curr_cl)], device=device)
        loss_fn = BalancedCrossEntropy(freq)


    # training loop (train/val)
    best_acc, best_model_state = 0.0, model.state_dict()
    acc_list = []

    print("\nTraining...")
    for epoch in range(EPOCHS):
        # training on current classes
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
                    "Step {} epoch {:2d} batch {:3d} [{:5d}/{:5d}] training loss: {:0.4f}".format(
                        step,
                        epoch,
                        batch_idx,
                        batch_idx * len(x),
                        len(train_loader.dataset),
                        loss.item(),
                    )
                )
        # eval on val set, current classes
        model.eval()
        correct = 0
        with torch.no_grad():
            confusion = torch.zeros(len(test_cl), len(test_cl))
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
        # keep the best model ckp
        if acc > best_acc:
            best_acc = acc
            best_model_state = copy.deepcopy(model.state_dict())
        print(
            "Step {} Val Acc: {:.2f}% ({}/{})".format(
                step, acc, correct, len(val_loader.dataset)
            )
        )
        torch.set_printoptions(sci_mode=False)
        # bonus : display confusion matrix
        # print(f"Confusion matrix at epoch {epoch}")

    print(f"Step {step} Val Acc history : {acc_list}")
    print("Best val acc {:.2f}".format(best_acc))

    # save best model
    ckp_name = "mnist_replay_incr_step{}_best.pt".format(step)
    ckp_path = os.path.join(ckp_root, ckp_name)
    print("\nSaving best model checkpoint under {}".format(ckp_path))
    torch.save(best_model_state, ckp_path)

    # eval on the cumulated test set (all classes seen so far)
    print("\nTesting ...")
    model.load_state_dict(torch.load(ckp_path, weights_only=True))
    model.eval()

    correct = 0
    with torch.no_grad():
        confusion = torch.zeros(len(test_cl), len(test_cl))
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            prediction = out.argmax(
                dim=1, keepdim=True
            )  # index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            for i, j in zip(prediction, target):
                confusion[i.to("cpu"), j.to("cpu")] += 1
    test_acc = 100.0 * correct / len(test_loader.dataset)
    test_acc_list.append(test_acc)
    nb_test_samples.append(len(test_loader.dataset))
    print(
        "Test Acc: {:.2f}% ({}/{})".format(test_acc, correct, len(test_loader.dataset))
    )
    torch.set_printoptions(sci_mode=False)
    print("Confusion matrix on test set")
    print(confusion.int().numpy())

    # compute test acc separately on each subset of classes
    correct_per_class = np.diag(confusion.numpy())
    target_per_class = np.sum(confusion.numpy(), axis=0)
    for k in range(step + 1):
        if k == 0:
            acc_mat[step][0] = np.sum(correct_per_class[:nb_init_cl]) / np.sum(
                target_per_class[:nb_init_cl]
            )
        else:
            test_acc_k = np.sum(
                correct_per_class[
                    nb_init_cl + (k - 1) * nb_incr_cl : nb_init_cl + k * nb_incr_cl
                ]
            ) / np.sum(
                target_per_class[
                    nb_init_cl + (k - 1) * nb_incr_cl : nb_init_cl + k * nb_incr_cl
                ]
            )
            acc_mat[step][k] = test_acc_k
    print(f"Test acc per group at step {step}: {acc_mat[step]}")


# Recap
print("\n=== Report - Replay ===\n")
print(
    "Scenario : {} initial classes + {} steps x {} classes = {} classes in total".format(
        nb_init_cl, nb_steps, nb_incr_cl, nb_tot_cl
    )
)
print(
    "\nHyperparameters : epochs {}, lr {}, archi CNN {}/{}/{}, replay with max_size {}, cumul {}".format(
        EPOCHS, lr, size_conv_1, size_conv_2, size_fc, max_size, cumul
    )
)

print("\nTest acc list : {}".format(["%.2f" % item for item in test_acc_list]))
print(">> Final acc {:.2f} <<".format(test_acc))

print("\nAccuracy matrix over the steps")
print(100 * acc_mat.round(2))

# avg incr acc / forgetting
avg_incr_acc = np.mean(test_acc_list)
avg_f = 100 * compute_forgetting(acc_mat[:,:-1] )
print("\nAvg incr acc: {:.2f}".format(avg_incr_acc))
print("MACRO Avg forgetting: {:.2f}".format(avg_f))

data = {
    "method" : ["replay"],
    "nb_init_cl" : [nb_init_cl],
    "nb_incr_cl" : [nb_incr_cl],
    "nb_tot_cl" : [nb_tot_cl],
    "memory" : [max_size], 
    "CE_loss" : [CE_type],
    "KD_loss" : ["None"],
    "lr" : [lr], 
    "epochs" : [EPOCHS],
    "last_acc" : [test_acc],
    "avg_incr_acc" : [avg_incr_acc],
    "avg_f" : [avg_f]
}
df1 = pd.DataFrame.from_dict(data)
print(df1)
df2 = pd.read_csv('logs/results.csv')
df2 = pd.concat([df2, df1], ignore_index=True)

df2.to_csv(os.path.join('logs', 'results.csv'), index=False)

elapsed = (time() - start) / 60  # elapsed time in minutes
print("\nCompleted expe in {:.2f} min".format(elapsed))
