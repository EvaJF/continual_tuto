import shutil
import torch
from multiprocessing import Pool
import time
import os
import sys
import json
import numpy as np
from configparser import ConfigParser
from utils_tuto.dataset import FeaturesDataset
import os
import torch
from torch import nn


class StreamingLDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant Analysis algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, test_batch_size=1024, shrinkage_param=1e-4,
                 streaming_update_sigma=True, device='cuda'):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param test_batch_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        """

        super(StreamingLDA, self).__init__()

        # SLDA parameters
        self.device = device
        #self.device = 'cuda'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():

            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

            # update class means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1

    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                #print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                        self.device))
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        #print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()
        #print('X shape: ', X.shape)
        #print('y shape: ', y.shape)
        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        #print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Sigma'] = self.Sigma.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_path, save_name):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_path, save_name + '.pth'))
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Sigma = d['Sigma'].to(self.device)
        self.num_updates = d['num_updates']


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def pool_feat(features):
    features = features[:, :, None , None]
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat

def save_predictions(y_pred, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'preds_min_trained_' + str(min_class_trained) + '_max_trained_' + str(max_class_trained) + suffix
    torch.save(y_pred, save_path + '/' + name + '.pt')

def save_accuracies(accuracies, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'accuracies_min_trained_' + str(min_class_trained) + '_max_trained_' + str(
        max_class_trained) + suffix + '.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))

def predict(model, val_data, device):
    samples = []
    labels = []
    for matrix, label in val_data:
        samples.append(matrix.numpy())
        labels.append(label.numpy())
    samples = np.concatenate(samples)
    labels = np.concatenate(labels)
    feat = torch.tensor(samples)
    feat = pool_feat(feat)
    labels = torch.tensor(labels)
    #probabilities = torch.empty((num_samples, current_last_class))
    probabilities = model.predict(feat.to(device), return_probas=True)
    return probabilities, labels

def compute_accuracies(loader, classifier, device):
    probas, y_test_init = predict(classifier, loader, device)
    top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
    return probas, y_test_init, top1, top5