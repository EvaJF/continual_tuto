import shutil
import torch
from multiprocessing import Pool
import time
import os
import numpy as np
import torch
from functools import partial
from time import time 

from methods.slda import (
    StreamingLDA,
    compute_accuracies,
    pool_feat,
    save_accuracies,
    save_predictions,
    predict,
    accuracy,
)
from utils_tuto.dataset import FeaturesDataset, untie_features, clean_features
from utils_tuto.parser import parse_args

# python dslda_expe.py --dataset flowers102 --nb_init_cl 52 --nb_incr_cl 10 --nb_tot_cl 102 --proj_dim 10000

if __name__ == "__main__":

    ### Params ###
    args = parse_args()
    batch_size = 32  # 1024
    num_workers = 8

    # data
    dataset = args.dataset
    nb_init_cl = args.nb_init_cl  # nbr of initial classes
    nb_incr_cl = args.nb_incr_cl  # nbr of new classes per increment
    nb_tot_cl = args.nb_tot_cl  # total number of classes
    n_tasks = (nb_tot_cl - nb_init_cl) // nb_incr_cl  # number of incremental steps
    assert nb_incr_cl == int(
        (nb_tot_cl - nb_init_cl) / n_tasks
    )  # nb classes in each incr. state
    min_state = int(nb_init_cl / nb_incr_cl - 1)
    max_state = int(nb_tot_cl / nb_incr_cl)
    print("nb_init_cl", nb_init_cl)
    print("nb_incr_cl", nb_incr_cl)
    print("nb_tot_cl", nb_tot_cl)
    print("min_state", min_state)
    print("max_state", max_state)

    # paths
    features_dir = args.features_dir
    log_dir = os.path.join(args.log_dir, args.archi, f"proj_{args.proj_dim}")
    batch_size = 32  # TODO augment batch size ?bigmem
    print("batch_size", batch_size)
    # feature path
    train_dir = os.path.join(features_dir, dataset, "train", f"proj_{args.proj_dim}")
    test_dir = os.path.join(features_dir, dataset, "test", f"proj_{args.proj_dim}")
    print("train_dir", train_dir)
    print("test_dir", test_dir)

    # model
    regul = 1.0
    toler = 0.0001
    random_seed = 42
    pretrained_net = args.archi
    if args.proj_dim > 0 : 
        feature_size = args.proj_dim
    else : 
        if pretrained_net == "resnet18":
            feature_size = 512
        elif pretrained_net == "resnet50" or pretrained_net == "trex":
            feature_size = 2048
        elif "vits-" in pretrained_net or "dino" in pretrained_net:
            feature_size = 384
        else:  # default
            raise NotImplementedError("Please provide a valid encoder name")

    print("pretrained_net", pretrained_net)
    pred_root = os.path.join(log_dir, "preds_slda")

    log_file = os.path.join(
        log_dir,
        f"b{nb_init_cl}/t{n_tasks}/slda.txt",
    )
    root_path_forgetting = os.path.join(
        pred_root, pretrained_net, dataset, f"b{nb_init_cl}", f"s{n_tasks}", "slda"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # compute confusion matrix
    print(
        f"b{nb_init_cl}/t{n_tasks}/slda_matconfusion.npy"
    )
    if not os.path.exists(
        f"{log_dir}/b{nb_init_cl}/t{n_tasks}/slda_matconfusion.npy"
    ):
        decompose_fn = partial(
            untie_features, train_features_path=train_dir, val_features_path=test_dir
        )
        print("\nStarting decomposition...")
        start = time()
        with Pool() as p:
            p.map(decompose_fn, range(nb_tot_cl))
        print("Decomposition completed in", time() - start, "seconds.")

        current_last_class = nb_init_cl

        save_dir = os.path.join(
            pred_root, "slda", dataset, "seed" + str(random_seed), "b" + str(nb_init_cl)
        )
        os.makedirs(save_dir, exist_ok=True)

        # setup SLDA model
        classifier = StreamingLDA(
            feature_size,
            nb_tot_cl,
            test_batch_size=batch_size,
            shrinkage_param=1e-4,
            streaming_update_sigma=False,
            device=device,
        )

        # run the streaming experiment
        start_time = time()
        # start list of accuracies
        accuracies = {"seen_classes_top1": [], "seen_classes_top5": []}

        first_time = True  # true for init step
        slda_save_name = "slda_model_weights_min_trained_0_max_trained_%d"
        init_classes = nb_init_cl
        class_increment = nb_incr_cl
        nb_tot_cl = nb_tot_cl
        # loop over all data and compute accuracy after every "batch"
        total_dict = []
        for curr_class_ix in [0] + list(
            range(init_classes, nb_tot_cl, class_increment)
        ):
            total_dict.append([])
            print("curr_class_ix", curr_class_ix)
            max_class = curr_class_ix + class_increment
            current_last_class = max_class
            if max_class == class_increment:
                max_class = init_classes
            print("\nTraining classes from {} to {}".format(curr_class_ix, max_class))
            # get training loader for current batch
            feat_train_dataset = FeaturesDataset(
                train_dir, range_classes=range(curr_class_ix, max_class)
            )
            feat_train_loader = torch.utils.data.DataLoader(
                feat_train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
            )
            # get the nubmer of training samples
            num_train_samples = len(feat_train_dataset)
            print(">>>>> size of train data:",num_train_samples)
            # convert it to a numpy array of samples, and a numpy array of labels, knowing that it yields tuples of (sample, label)
            samples = []
            labels = []
            for matrix, label in feat_train_loader:
                samples.append(matrix.numpy())
                labels.append(label.numpy())
            samples = np.concatenate(samples)
            labels = np.concatenate(labels)
            # shuffle the data # TODO remove / replace by shuffle=True in Dataloader
            perm = np.random.permutation(len(labels))
            labels = labels[perm]
            samples = samples[perm]

            if first_time:
                print('\nGetting data for model initialization...')
                base_init_data, base_init_labels = (
                    samples,
                    labels,
                ) 
                base_init_data = torch.tensor(base_init_data)
                base_init_labels = torch.tensor(base_init_labels)
                print('\nFitting initial model...')
                classifier.fit_base(base_init_data, base_init_labels)
                first_time = False
            else:
                # fit model
                batch_x_feat = torch.tensor(samples)
                batch_y = torch.tensor(labels)
                batch_x_feat = pool_feat(batch_x_feat)
                for x, y in zip(batch_x_feat, batch_y):
                    classifier.fit(
                        x.cpu(),
                        y.view(
                            1,
                        ),
                    )

            # output accuracies to console and save out to json file
            seen_classes_test_dataset = FeaturesDataset(
                test_dir, range_classes=range(max_class)
            )
            seen_classes_test_loader = torch.utils.data.DataLoader(
                seen_classes_test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )
            # get_data_loader(train_file_path, val_file_path, dataset_name, datasets_mean_std_file_path, False, 0, curr_max_class, batch_size=batch_size,shuffle=shuffle)
            seen_probas, y_test, seen_top1, seen_top5 = compute_accuracies(
                seen_classes_test_loader, classifier, device
            )
            confusion_matrix = torch.zeros(max_class, max_class)
            # print(seen_probas, y_test)
            # prune the seen classes
            y_pred = seen_probas[:, :max_class].argmax(dim=1, keepdim=True)
            # print(y_test.view(-1), y_pred.view(-1))
            for t, p in zip(y_test.view(-1), y_pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            confusion_matrix = confusion_matrix.numpy()
            for c in range(max_class):
                total_dict[-1].append(
                    confusion_matrix[c, c].item() / confusion_matrix[c, :].sum()
                )
            # print(total_dict[-1])
            # print(confusion_matrix)
            os.makedirs(root_path_forgetting, exist_ok=True)
            np.save(
                root_path_forgetting
                + "dict_per_class_acc_maxclass"
                + str(max_class)
                + ".npy",
                total_dict[-1],
            )
            np.save(
                root_path_forgetting
                + "confusion_matrix_maxclass"
                + str(max_class)
                + ".npy",
                confusion_matrix,
            )

            curr_max_class = max_class
            print(
                "Seen Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%"
                % (0, curr_max_class - 1, seen_top1, seen_top5)
            )
            accuracies["seen_classes_top1"].append(float(seen_top1))
            accuracies["seen_classes_top5"].append(float(seen_top5))

            # save accuracies and predictions out
            save_accuracies(
                accuracies,
                min_class_trained=0,
                max_class_trained=curr_max_class,
                save_path=save_dir,
            )
            save_predictions(seen_probas, 0, curr_max_class, save_dir)
            top5_acc = np.array(accuracies["seen_classes_top5"])
            top1_acc = np.array(accuracies["seen_classes_top1"])

            # print('TOP1 accuracies so far = {}'.format(str(top1_acc)))
            # print('TOP5 accuracies so far = {}'.format(str(top5_acc)))
            mean_top1_acc = np.mean(top1_acc)
            # print('MEAN TOP1:',mean_top1_acc)

            #classifier.save_model(save_dir, slda_save_name % max_class)

        # print final accuracies and time
        test_dataset = FeaturesDataset(test_dir, range_classes=range(nb_tot_cl))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        probas, y_test = predict(classifier, test_loader, device)
        top1, top5 = accuracy(probas, y_test, topk=(1, 5))
        #classifier.save_model(save_dir, "slda_model_weights_final")
        end_time = time()
        print("\nFinal: top1=%0.2f%% -- top5=%0.2f%%" % (top1, top5))
        print("\nTotal Time (seconds): %0.2f" % (end_time - start_time))

        print("****************************************")
        top5_acc = np.array(accuracies["seen_classes_top5"])
        top1_acc = np.array(accuracies["seen_classes_top1"])

        print("TOP1 accuracies = {}".format(str(top1_acc)))
        print("TOP5 accuracies = {}".format(str(top5_acc)))

        print("Mean TOP1 accuracy = {:.5f}".format(np.mean(top1_acc[1:])))
        print("Mean TOP5 accuracy = {:.5f}".format(np.mean(top5_acc[1:])))

        top1 = top1_acc
        top5 = top5_acc

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(
                f"======{dataset} seed{random_seed} b{nb_init_cl} t{n_tasks}======\n"
            )
            f.write("top1:" + str([round(elt, 2) for elt in top1]) + "\n")
            f.write("top5:" + str([round(elt, 2) for elt in top5]) + "\n")
            # f.write("top1_past:"+str([round(100*elt,2) for elt in top1_past])+"\n")
            # f.write("top1_current:"+str([round(100*elt,2) for elt in top1_current])+"\n")
            f.write(
                f"top1 = {sum(top1)/len(top1):.3f}, top5 = {sum(top5)/len(top5):.3f} | top1 without first batch = {sum(top1[1:])/len(top1[1:]):.3f}, top5 without first batch = {sum(top5[1:])/len(top5[1:]):.3f} \n"
            )
            f.write(
                "=======================================================================\n"
            )

        mat_forgetting = []
        for ms in range(min_state, max_state + 1):
            mat_forgetting.append([])
            list_forgetting = []

            for c in range(nb_tot_cl):
                list_acc_c = []
                for s in range(min_state, ms + 1):
                    try:
                        list_acc_c.append(total_dict[s - min_state][c])
                    except:
                        list_acc_c.append(0)
                forgetting = list_acc_c[-1] - max(list_acc_c)
                list_forgetting.append(-forgetting)
                # print(f"Class {c} forgetting {forgetting}, {list_acc_c}")
            # save the forgetting list in a file
            np.save(
                root_path_forgetting + f"list_forgetting_state{ms-min_state}.npy",
                list_forgetting,
            )
            # print(f"Mean forgetting for {pretrained_net} {dataset} b{nb_init_cl} s{n_tasks} - state {ms-min_state} is {round(np.mean(list_forgetting), 3)}")
            for batch in range(min_state, ms + 1):
                if min_state > 0:
                    if batch == min_state:
                        coucou = list_forgetting[0 : (batch + 1) * nb_incr_cl]
                    else:
                        coucou = list_forgetting[
                            batch * nb_incr_cl : (batch + 1) * nb_incr_cl
                        ]
                else:
                    coucou = list_forgetting[
                        batch * nb_incr_cl : (batch + 1) * nb_incr_cl
                    ]
                # print(f"Batch {batch} forgetting {round(np.mean(coucou), 3)}")
                mat_forgetting[-1].append(round(np.mean(coucou), 3))

        # mat_forgetting = np.array(mat_forgetting)
        for i in range(len(mat_forgetting)):
            # 0-padding to have a square matrix
            mat_forgetting[i] = np.pad(
                mat_forgetting[i],
                (0, len(mat_forgetting[-1]) - len(mat_forgetting[i])),
                "constant",
                constant_values=(0, 0),
            )
        mat_forgetting = np.array(mat_forgetting)
        print(mat_forgetting)
        np.save(root_path_forgetting + f"mat_forgetting.npy", mat_forgetting)
        # /home/data/efeillet/ISmaIL/logs/byol-resnet50-imagenet100/casia-align/seed-1/b10/t9/slda_matforgetting.npy
        os.makedirs(
            f"{log_dir}/b{nb_init_cl}/t{n_tasks}",
            exist_ok=True,
        )
        np.save(
            f"{log_dir}/b{nb_init_cl}/t{n_tasks}/slda_matforgetting.npy",
            mat_forgetting,
        )
        print(
            "saved in ",
            f"{log_dir}/b{nb_init_cl}/t{n_tasks}/slda_matforgetting.npy",
        )
        # copy root_path_forgetting + "confusion_matrix_maxclass100.npy" in f'/home/data/efeillet/ISmaIL/logs/b{nb_init_cl}/t{n_tasks}/slda_matconfusion.npy'
        shutil.copy(
            root_path_forgetting + f"confusion_matrix_maxclass{nb_tot_cl}.npy",
            f"{log_dir}/b{nb_init_cl}/t{n_tasks}/slda_matconfusion.npy",
        )

    try:
        # display log_file content
        print(
            "=====================================",
            log_file,
            "=====================================",
        )
        with open(log_file, "r") as f:
            print(f.read())
    except:
        print("Not yet computed")

    clean_fn = partial(
        clean_features, train_features_path=train_dir, val_features_path=test_dir
    )
    with Pool(8) as p:
        p.map(clean_fn, range(nb_tot_cl))

    print("Done cleaning")