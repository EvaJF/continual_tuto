import torch
from torch.nn import functional as F
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from time import time
from functools import partial

from utils_tuto.dataset import FeaturesDataset, untie_features, clean_features
from utils_tuto.parser import parse_args

# Example usage
# python fecam1_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102

### Utility functions ###


def tukeys_transform(x, beta=0.5):
    x = torch.relu(x)
    if beta == 0:
        return torch.log(x)  # TODO compare with just x for dino
    else:
        return torch.pow(x, beta)


def shrink_cov(cov, alpha1, alpha2):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.detach().clone()
    torch.diagonal(off_diag).fill_(
        0.0
    )  # torch 2.2 : off_diag.fill_diagonal(0.0) # np.fill_diagonal(off_diag,0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0])
    cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
    return cov_


def normalize_cov(cov):
    sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
    cov = cov / (torch.matmul(torch.unsqueeze(sd, 1), torch.unsqueeze(sd, 0)))
    return cov


def mahalanobis(dist, inv_covmat):
    # inv_covmat = torch.linalg.pinv(cov)
    left_term = torch.matmul(dist, inv_covmat)
    mahal = torch.matmul(left_term, dist.T)
    return torch.diagonal(mahal, 0)


def Frobenius(M):
    return torch.sqrt(torch.sum(M**2))


def main():

    ### Params ###
    args = parse_args()

    # data
    dataset = args.dataset
    nb_init_cl = args.nb_init_cl  # nbr of initial classes
    nb_incr_cl = args.nb_incr_cl  # nbr of new classes per increment
    nb_tot_cl = args.nb_tot_cl  # total number of classes
    n_tasks = (nb_tot_cl - nb_init_cl) // nb_incr_cl  # number of incremental steps

    # paths
    features_dir = args.features_dir
    log_dir = args.log_dir
    batch_size = 32  # TODO augment batch size ?bigmem
    print("batch_size", batch_size)
    # feature path
    train_dir = os.path.join(features_dir, dataset, "train")
    test_dir = os.path.join(features_dir, dataset, "test")
    print("train_dir", train_dir)
    print("test_dir", test_dir)
    # TODO fix scratch
    scratch_train_dir = train_dir
    scratch_test_dir = test_dir
    print("scratch_train_dir", scratch_train_dir)
    print("scratch_test_dir", scratch_test_dir)

    # feature size
    feature_extractor = args.archi
    if feature_extractor == "resnet18":
        hidden_size = 512
    elif feature_extractor == "resnet50":
        hidden_size = 2048
    elif (
        "vits" in feature_extractor or feature_extractor == "dinov2"
    ):  # by default, dinov2 --> dinov2-vits-imagenet2012
        hidden_size = 384
    else:  # ViT base
        hidden_size = 768
    print("feature size :", hidden_size)

    ### Open features ###
    decompose_fn = partial(
        untie_features, train_features_path=train_dir, val_features_path=test_dir
    )

    print("\nStarting decomposition...")
    start = time()
    with Pool() as p:
        p.map(decompose_fn, range(nb_tot_cl))
    print("Decomposition completed in", time() - start, "seconds.")

    ### Training ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\nBegining training with device", device)
    task_size_train = torch.zeros(n_tasks + 1)
    class_mean_set = []
    accuracy_history_u = []

    for task in range(n_tasks + 1):
        print(f"\nTask {task}")
        if task == 0:
            start_id, stop_id = 0, nb_init_cl
        else:
            start_id = nb_init_cl + nb_incr_cl * (task - 1)
            stop_id = start_id + nb_incr_cl
        print(f"Train index : {start_id} - {stop_id}")
        print(f"Test  index : 0 - {stop_id}")
        train_dataset = FeaturesDataset(
            scratch_train_dir, range_classes=range(start_id, stop_id)
        )
        test_dataset = FeaturesDataset(
            scratch_test_dir, range_classes=range(0, stop_id)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        X, y = [], []
        for img_batch, label in tqdm(
            train_loader, desc=f"Training {task}", total=len(train_loader)
        ):  # train_loader
            img_batch.to(device)
            label.to(device)
            out = F.normalize(img_batch.clone().detach())  # .numpy()
            X.append(out)
            y.append(label)
        X = torch.cat(
            X
        )  # NB : bizarre de normaliser par batch sachant qu'on a X en mémoire, mais peut-être que c'est plus réaliste par rapport à l'inférence ?
        y = torch.cat(y)
        task_size_train[task] += len(X)
        print(f"Train size of task {task} is {len(X)}")

        # Compute the class prototypes of the current classes
        for i in range(start_id, stop_id):  # bottleneck 1
            # select the features of the current class
            image_class_mask = y == i
            # compute the class prototype
            class_mean_set.append(torch.mean(X[image_class_mask], axis=0))  # np.mean

        # Compute the current covariance matrix
        curr_cov = torch.cov(X.T)  # np.cov
        # print(f"Frobenius of current cov mat {task}    : {Frobenius(curr_cov)}")

        # Update the common covariance matrix
        if task == 0:  # initial task, only 1 cov mat
            common_cov = curr_cov.detach().clone()
            if ("vits" in feature_extractor) or (feature_extractor == "dinov2"):
                alpha1, alpha2 = 0.0, 0.0
            else:
                alpha1, alpha2 = 10.0, 10.0
        else:  # weighted average with the previous matrix
            ratio = torch.sum(task_size_train[:task]) / torch.sum(task_size_train)
            # print('Update ratio', ratio)
            common_cov = ratio * common_cov + (1 - ratio) * curr_cov  # equation (6)
            # print(f"Frobenius of COMMON cov mat {task} : {Frobenius(common_cov)}")
            # print(f"Frobenius of DELTA cov mat {task}  : {Frobenius(common_cov - curr_cov)}")
            alpha1, alpha2 = 0.0, 0.0  # by default, no shrinkage for incremental tasks

        # shrink before inversion
        print("alpha1, alpha2 :", alpha1, alpha2)
        if alpha1 == 0.0 and alpha2 == 0.0:
            print("no shrinkage")
            shrunk_common_cov = common_cov
        else:
            shrunk_common_cov = shrink_cov(common_cov, alpha1=alpha1, alpha2=alpha2)
        # print(f"Frobenius of SHRUNK cov mat {task}     : {Frobenius(shrunk_common_cov)}")
        # print(f"Frobenius of DELTA SHRUNK cov mat {task} : {Frobenius(common_cov - shrunk_common_cov)}")
        # Invert the common cov mat with Moore-Penrose algorithm
        pinv_common_cov = torch.linalg.pinv(shrunk_common_cov)

        print("\nInference")
        correct_u, total_u = 0, 0

        for img_batch, label in tqdm(
            test_loader, desc=f"Testing {task}", total=len(test_loader)
        ):  # test_loader :
            img_batch.to(device)
            label.to(device)
            out = F.normalize(img_batch.clone().detach())  # .numpy()

            predictions_u = []  # v2 : unique covariance matrix
            maha_dist_u = []

            for cl in range(
                stop_id
            ):  # bottleneck 2 TODO optimize : compute everything only once ?
                distance_cl = out - class_mean_set[cl]
                # UNIQUE cov
                dist_u = mahalanobis(distance_cl, pinv_common_cov)
                maha_dist_u.append(dist_u.numpy())  ### back to numpy from here

            maha_dist_u = np.array(maha_dist_u)
            pred_u = np.argmin(maha_dist_u.T, axis=1)
            predictions_u.append(pred_u)

            predictions_u = torch.tensor(np.array(predictions_u))
            correct_u += (predictions_u.cpu() == label.cpu()).sum()
            total_u += label.shape[0]

        print(f"Accuracy at {task} with UNIQUE cov matrix {correct_u/total_u}")
        accuracy_history_u.append(float(correct_u / total_u))

    print(f"\nCOMMON COV Average Incremental Accuracy: {np.mean(accuracy_history_u)}")

    ### print report ###
    log_path = os.path.join(
        log_dir, feature_extractor, dataset, "seed-1", f"b{nb_init_cl}", f"t{n_tasks}"
    )
    print("Writing log to", log_path)
    os.makedirs(log_path, exist_ok=True)

    # compute avg incr acc
    print("\nRecap Acc COMMON COV")
    print("Avg acc list ", [round(k, 3) for k in accuracy_history_u])
    avg_incr_acc_u = round(np.mean(accuracy_history_u), 3)
    print("Avg incr acc", avg_incr_acc_u)
    log = f"""=========== {dataset} seed-1 b{nb_init_cl} t{n_tasks} ==============
top1:{accuracy_history_u}
top1 = {avg_incr_acc_u} | top1 without first batch = {round(np.mean(accuracy_history_u[1:]),3)}
================================================================
"""
    print(log)
    with open(os.path.join(log_path, "fecam_common.txt"), "w") as f:
        f.write(log)
    print("Wrote log")

    clean_fn = partial(
        clean_features, train_features_path=train_dir, val_features_path=test_dir
    )
    with Pool(8) as p:
        p.map(clean_fn, range(nb_tot_cl))


if __name__ == "__main__":
    main()
