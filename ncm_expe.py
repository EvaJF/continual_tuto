import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils_tuto.dataset import read_features_as_array
from utils_tuto.parser import parse_args
from methods.ncm import NCM_incr

# Example usage
# python ncm_expe.py --dataset mnist --nb_tot_cl 10 --nb_init_cl 2 --nb_incr_cl 2 --archi simpleCNN --pretrain none
# python ncm_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102 --archi vits --pretrain lvd142m

# params
args = parse_args()

dataset = args.dataset
nb_init_cl = args.nb_init_cl
nb_incr_cl = args.nb_incr_cl
nb_tot_cl = args.nb_tot_cl
n_steps = (nb_tot_cl - nb_init_cl) // nb_incr_cl
print(n_steps, "steps")

log_dir = os.path.join(args.log_dir, args.archi)
prefix = args.prefix
features_dir = args.features_dir  
train_dir = os.path.join(features_dir, dataset, args.archi, args.pretrain, "train")
test_dir = os.path.join(features_dir, dataset, args.archi, args.pretrain, "test")

print(f"Loading features from {train_dir}")
X_train, y_train = read_features_as_array(train_dir, nb_tot_cl)
print(f"Loading features from {train_dir}")
X_test, y_test = read_features_as_array(test_dir, nb_tot_cl)
print("Data shape", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# standard scaling
print("\nPreprocessing")
X_init = X_train[y_train < nb_init_cl]
std_scaler = StandardScaler().fit(X_init)
X_train, X_test = std_scaler.transform(X_train), std_scaler.transform(X_test)

# model initialization
print("\nInit step")
ncm = NCM_incr()
ncm.fit(X_train[y_train < nb_init_cl], y_train[y_train < nb_init_cl])
y_pred = ncm.predict(X_test[y_test < nb_init_cl])
acc = accuracy_score(y_test[y_test < nb_init_cl], y_pred)
print(acc)
avg_acc_list, curr_acc_list, init_acc_list = [], [], []
avg_acc_list.append(acc)
curr_acc_list.append(acc)
init_acc_list.append(acc)

# incremental steps
for s in range(0, n_steps):
    print("\nIncr step", s)
    start_idx, stop_idx = nb_init_cl + nb_incr_cl * s, nb_init_cl + nb_incr_cl * (s + 1)
    print("start_idx, stop_idx", start_idx, stop_idx)
    index_s = np.concatenate(
        [np.where(y_train == i) for i in range(start_idx, stop_idx)], axis=1
    ).ravel()
    print("index train ", index_s)
    y_train_s = y_train[index_s]
    print("labels train", y_train_s)
    X_train_s = X_train[index_s]
    print("y_train_s.shape", y_train_s.shape)
    print("X_train_s.shape", X_train_s.shape)
    print(y_train_s.max(), ncm.n_classes, len(np.unique(y_train_s)))
    ncm.update(X_train_s, y_train_s)
    # test acc on cumulated test set
    y_test_s = y_test[y_test < stop_idx]
    X_test_s = X_test[y_test < stop_idx]
    print("y_test_s.shape", y_test_s.shape)
    print("X_test_s.shape", X_test_s.shape)
    y_pred_s = ncm.predict(X_test_s)
    acc_s = accuracy_score(y_test_s, y_pred_s)
    print(acc_s)
    avg_acc_list.append(acc_s)
    # test acc on current classes only
    index_curr = np.concatenate(
        [np.where(y_test == i) for i in range(start_idx, stop_idx)], axis=1
    ).ravel()
    y_test_curr = y_test[index_curr]
    print("y_test_curr", y_test_curr.shape)
    X_test_curr = X_test[index_curr]
    y_pred_curr = ncm.predict(X_test_curr)
    acc_curr = accuracy_score(y_test_curr, y_pred_curr)
    curr_acc_list.append(acc_curr)
    # test acc on initial classes only
    y_pred_init = ncm.predict(X_test[y_test < nb_init_cl])
    acc_init = accuracy_score(y_test[y_test < nb_init_cl], y_pred_init)
    init_acc_list.append(acc_init)

# compute avg incr acc
print("\nRecap Acc")
print("Avg acc list ", [round(k, 3) for k in avg_acc_list])
print("Curr acc list", [round(k, 3) for k in curr_acc_list])
print("Init acc list", [round(k, 3) for k in init_acc_list])
avg_incr_acc = round(np.mean(avg_acc_list), 3)
print("Avg incr acc", avg_incr_acc)
# save ncm.txt with top1 acc in the same format as fetril and slda
log = f"""=========== {prefix} seed-1 b{nb_init_cl} t{n_steps} ==============
top1:{avg_acc_list}
top1_curr:{curr_acc_list}
top1_init:{init_acc_list}
top1 = {avg_incr_acc} | top1 without first batch = {round(np.mean(avg_acc_list[1:]),3)}
================================================================
"""
print(log)
log_path = os.path.join(log_dir, f"b{nb_init_cl}", f"t{n_steps}")
print("Writing log to", log_path)
os.makedirs(log_path, exist_ok=True)
with open(os.path.join(log_path, "ncm.txt"), "w") as f:
    f.write(log)
print("Wrote log")
