#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:19:41 2021

@author: Aakash
"""
from strokenet import StrokeNet
from skorch import NeuralNetClassifier
import torch
from utils import StrokeDataset
from sklearn.metrics import make_scorer
import numpy as np


def accuracy_argmax(y_true, y_pred):
    return np.mean(y_true = np.argmax(y_pred, -1))


def train_strokenet():
    model = StrokeNet(num_conv_layers=0, seq_len=240, input_dim=5664, 
                      pool='fc')
    lr = 3e-4
    root_path = 'some_path'
    accuracy_argmax_scorer = make_scorer(accuracy_argmax)
    net = NeuralNetClassifier(model,
                              optimizer=torch.optim.Adam(model.parameters(),
                                                         lr=lr),
                              max_epochs=10, batch_size=2,
                              criterion=torch.nn.CrossEntropyLoss(),
                              iterator_train__shuffle=True,
                              dataset=StrokeDataset(root_path=root_path,
                                                    seq_len=240,
                                                    drop_no_patient=True))