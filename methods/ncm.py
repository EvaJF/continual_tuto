import os
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class NCM_incr:
    """
    Perform nearest class mean classification.
    """

    def __init__(self, distance="cosine"):
        self._proto_dict = {}
        self.n_classes = 0
        self.prototypes = np.array([])

        if distance == "euclidean":
            self.distance = euclidean_distances
        elif distance == "cosine":
            self.distance = cosine_distances

    def fit(self, X, y):
        labels = list(np.unique(y))
        self.n_classes = len(labels)
        print(min(labels), max(labels))

        # compute prototypes
        for i in labels:
            # select samples from class i only
            X_i = X[y == i]
            # print(f"Class {i}, X_i : {X_i.shape}")
            p_i = np.mean(X_i, axis=0).reshape((1, -1))
            self._proto_dict[i] = p_i
            # print("prototype", p_i.shape)

        self.prototypes = np.array(
            [self._proto_dict[i].ravel() for i in range(self.n_classes)]
        )
        # print("P shape", self.prototypes.shape)

        return None

    def update(self, X, y):
        """
        new set of data to take into account
        hyp : new classes only
        """
        new_labels = list(np.unique(y))
        print("new labels", new_labels)
        max_label = max(y)
        assert self.n_classes + len(new_labels) == max_label + 1
        self.n_classes = max_label + 1

        # compute prototypes
        for i in new_labels:
            # select samples from class i only
            X_i = X[y == i]
            # print(f"Class {i}, X_i : {X_i.shape}")
            p_i = np.mean(X_i, axis=0).reshape((1, -1))
            self._proto_dict[i] = p_i
            # print("prototype", p_i.shape)
        self.prototypes = np.array(
            [self._proto_dict[i].ravel() for i in range(self.n_classes)]
        )
        print("P shape", self.prototypes.shape)
        return None

    def predict(self, X, batch_size=256):  # 128
        print("predict - len proto", len(self.prototypes))
        print("predict - len data ", len(X))
        assert len(self.prototypes) == self.n_classes
        n_samples = X.shape[0]
        # compute predictions
        Y_pred = []
        ##print("n_batches", n_samples//batch_size+1)
        for b in range(max(1, n_samples // batch_size + 1)):
            samples = X[batch_size * b : batch_size * (b + 1)]
            if len(samples) == 0:
                print("empty slice", b, n_samples, batch_size)
            else:
                D = self.distance(samples, self.prototypes)
                ##print(D.shape)
                pred = np.argmin(D, axis=1)  # .reshape((-1, 1))
                ##print(pred.shape)
                Y_pred.append(pred.reshape(1, -1))
        Y_pred = np.concatenate(Y_pred, axis=1)
        Y_pred = Y_pred.reshape((Y_pred.shape[1],))
        assert len(Y_pred) == n_samples
        return Y_pred

    def nearest_confounding(self, X, y):
        """
        Compute distance to nearest confounding class.
        """
        D_sample = []  # distance to the nearest confounding class for each sample
        D_avg = []  # average results per class
        # for each class, select the samples X_i
        # and take all prototypes except the one from the class
        for i in range(self.n_classes):
            X_i = X[y == i]
            # print(f'Class {i}, {len(X_i)}')
            if len(X_i) == 0:
                D_avg.append(0.0)
            else:
                P_i = self.prototypes[
                    [k for k in range(len(self.prototypes)) if k != i]
                ]
                # print("P_i.shape", P_i.shape)
                assert P_i.shape[0] == self.n_classes - 1
                D = self.distance(X_i, P_i)
                # print("D.shape", D.shape)
                nearest = np.min(D, axis=1)
                # print("nearest.shape", nearest.shape)
                D_sample += [e for e in nearest]
                D_avg.append(np.mean(nearest))
        # print("len(D_sample)", len(D_sample))
        # print("len(D_avg)", len(D_avg))
        return D_sample, D_avg
