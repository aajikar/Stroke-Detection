# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:54:39 2021

@author: BTLab
"""
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
from pandas.errors import MergeError
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from strokenet import StrokeNet, StrokeNetV2, StrokeInceptionNet
from joblib import Parallel, delayed
import functools
import time
import os
from torch.optim.lr_scheduler import StepLR
from i3d import InceptionI3d
import scipy.ndimage
from RollingWindow import CreateRollingWindow as rw

def sliding_window(data, seq_length, label):
    xs = []
    ys = [] # This will just be the actual label, will know in advance
    
    # Get the first sequnce in large timeseries, then move seq_length and get
    # The next sequence, e.g., Frames 0-119, Frames 120-239, etc.
    for i in range(len(data) - seq_length - 1):
        pass
    
    return np.array(xs), np.array(ys)


# TODO: Create a function that does a min-max scaling on the data
def min_max_scaler():
    return None

def timer(func):
        """Print the runtime of the decorated function."""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = start_time - end_time
            print(f"Finished {func.__name__!r} in {-run_time:.4f} s")
            return value
        return wrapper_timer

# TODO: Dataset class that can be initialized with different sequence length
# The data is saved as frames. The user specifies sequence length and samples
# get varied based on the length. Each sample is shape->(x, t) where x is the
# vectorized pressure frame data t is the seq length. Seq length may not always
# be the same, especially for the last sequence so need to add padding

# 1. Setup pytorch dataset to load the data
# 2. Setup padding of every batch (all samples must be of same seq_len)
# 3. Setup dataloader

# TODO: Separate continuous time series.
# The pressure data has abrupt discontinuties w.r.t. time. Seperate these large
# chunks of data into different locations (or find a way to annotate this)
# Do the sequence length dataset thing with these chunks


def test_model(model, dataset, log_dir, force_binary=False):
    batch_size = 1
    shuffle_dataset = True
    random_seed = 2021
    if force_binary:
        num_classes = 2
    else:
         num_classes = 3
    
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    test_sampler = SubsetRandomSampler(indices)
    
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=test_sampler, num_workers=16)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    test_scores = [{'Accuracy': pd.Series(),
                   'Precision': pd.Series(),
                   'Recall': pd.Series(),
                   'Specificity': pd.Series(),
                   'F1': pd.Series()}
                  for i in range(num_classes)]

    torch.cuda.empty_cache()

    test_results, y_true, y_pred = validation_step(model, test_loader, device, 
                                                   num_classes, epoch=1, 
                                                   writer=writer,
                                                   validation=False,
                                                   return_preds=True)

    return test_results, y_true, y_pred


def train_model(model, dataset, log_dir, k_fold=5, binary=False,
                validation_dataset=None, continue_training=False,
                current_epoch=None, num_epochs=25, save_every_epoch=False):
    # targets should be the label number (not one hot)
    # TODO: Make all the below parameters be changeable
    loss_fn = nn.CrossEntropyLoss()
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs = 50
    validation_split = 0.2
    batch_size = 1
    shuffle_dataset = True
    random_seed = 42
    if binary:
        num_classes = 2
    else:
        num_classes = 3
    
    step = 0
    device = \
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    train_params = {'Learning Rate': lr,
                    'Epochs': num_epochs,
                    'Batch Size': batch_size,
                    'Shuffle': shuffle_dataset,
                    'Random Seed': random_seed,
                    'Folds': k_fold}
    
    if k_fold == 1:
        # Create a train loader and validation loader
        # 80/20 split for now because LOSO will cause training data to be extremely
        # unbalanced when the sample size is small
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        

        train_loader = DataLoader(dataset,
                                  batch_size=batch_size, 
                                  sampler=train_sampler,
                                  num_workers=16)
        validation_loader = DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=valid_sampler,
                                       num_workers=16)
        # Update the current log dir for the k fold
                
        writer = SummaryWriter(log_dir=str(log_dir))

    elif k_fold == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        assert validation_dataset is not None, "A validation dataset must be manually specified when k_fold is equal to 0"
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        writer = SummaryWriter(log_dir=str(log_dir))

    else:
        total_size = len(dataset)
        
        fraction = 1/k_fold
        seg = int(total_size * fraction)

    train_scores = [{'Accuracy': pd.Series(),
                     'Precision': pd.Series(),
                     'Recall': pd.Series(),
                     'Specificity': pd.Series(),
                     'F1': pd.Series()}
                    for i in range(num_classes)]

    val_scores = [{'Accuracy': pd.Series(),
                   'Precision': pd.Series(),
                   'Recall': pd.Series(),
                   'Specificity': pd.Series(),
                   'F1': pd.Series()}
                  for i in range(num_classes)]

    torch.cuda.empty_cache()

    if k_fold == 0:
        k_fold += 1

    for i in range(k_fold):
        step = 0
        for param in model.parameters():
            param.requires_grad = True
            
        # Reset the optimizer every fold
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        if k_fold > 1:
            # Set up the folds
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_size
            
            # Update the current log dir for the k fold
            folder_name = "Fold" + str(i+1)
            current_log_dir = Path.joinpath(log_dir, folder_name)
            
            writer = SummaryWriter(log_dir=str(current_log_dir))
            
            train_left_indices = list(range(trll,trlr))
            train_right_indices = list(range(trrl,trrr))
            
            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall,valr))
            
            train_set = Subset(dataset,train_indices)
            val_set = Subset(dataset,val_indices)
            
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=16)
            validation_loader = DataLoader(val_set, batch_size=batch_size,
                                           shuffle=True, num_workers=16)
        

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1} of {num_epochs}\n")
            
            scheduler.step()
    
            # Perform training
            train_results, step = train_step(model, optimizer, loss_fn,
                                             train_loader, device, num_classes,
                                             epoch, writer, step)

            torch.cuda.empty_cache()
            
            # Perform validation
            valid_results = validation_step(model, validation_loader, device, 
                                            num_classes, epoch, writer)
            
            if save_every_epoch:
                if k_fold == 1 or k_fold == 0:
                    save_model(model, optimizer, train_params, epoch, log_dir)
                else:
                    save_model(model, optimizer, train_params, epoch, current_log_dir)
            
            torch.cuda.empty_cache()
            # Increment step
            step += 1
        
        # Get the final scores after all the epochs
        for j in range(num_classes):
            train_scores[j]['Accuracy'].at[i] = train_results[j]['accuracy']
            train_scores[j]['Precision'].at[i] = train_results[j]['precision']
            train_scores[j]['Recall'].at[i] = train_results[j]['recall']
            train_scores[j]['Specificity'].at[i] = train_results[j]['specificity']
            if not (0 in [train_results[j]['precision'] + train_results[j]['recall']]):
                train_scores[j]['F1'].at[i] = 2 * ((train_results[j]['precision'] *
                                                    train_results[j]['recall']) / 
                                                   (train_results[j]['precision'] +
                                                    train_results[j]['recall']))
            else:
                train_scores[j]['F1'].at[i] = 0
            
            val_scores[j]['Accuracy'].at[i] = valid_results[j]['accuracy']
            val_scores[j]['Precision'].at[i] = valid_results[j]['precision']
            val_scores[j]['Recall'].at[i] = valid_results[j]['recall']
            val_scores[j]['Specificity'].at[i] = valid_results[j]['specificity']
            if not (0 in [valid_results[j]['precision'] + valid_results[j]['recall']]):
                val_scores[j]['F1'].at[i] = 2 * ((valid_results[j]['precision'] *
                                                    valid_results[j]['recall']) / 
                                                   (valid_results[j]['precision'] +
                                                    valid_results[j]['recall']))
            else:
                val_scores[j]['F1'].at[i] = 0
        
        if k_fold == 1 or k_fold == 0:
            save_model(model, optimizer, train_params, num_epochs, log_dir)
        else:
            save_model(model, optimizer, train_params, num_epochs, current_log_dir)

        torch.cuda.empty_cache()
        
    return train_scores, val_scores, model


def train_step(model, optimizer, loss_function, train_loader, device,
               num_classes, epoch, writer, step):
    """
    Perform the training step in the epoch.

    Parameters
    ----------
    model : nn.Module
        Pytorch model.
    optimizer : torch.optim
        Optimizer used for training the model.
    loss_function : torch.loss
        Loss function used in the model.
    valid_loader : DataLoader
        DataLoader object containing the validation data.
    device : torch.device
        The device where the data is located.
    num_classes : int
        Number of outputs from the model.
    epoch : int
        Current elapsed epochs.
    writer : SummaryWriter
        Tensorboard SummaryWriter object.
    step : int
        Global step of the current sample.

    Returns
    -------
    train_results : list
        List of dictionary containing accuracy, precision, recall,
        and specificity for each class.
    step : int
        Updated global step of the current sample.

    """
    # First set the model into training mode
    model.train()
    print("\nTraining\n")
    
    # Create metrics prior to starting training loop
    metric = [{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'accuracy': 0,
               'precision': 0, 'recall': 0, 'specificity': 0}
              for i in range(num_classes)]
    
    
    for sample in tqdm(train_loader):
        sequence = sample['sequence']
        target = sample['label']
        if torch.cuda.is_available():
            model.cuda()
            sequence = sequence.cuda()
            target = target.cuda()
        y_pred = model(sequence)
        one_hot_target = []
        for i in range(sample['label'].shape[0]):
            one_hot_target.append(np.eye(num_classes)[sample['label'][i]])
        
        one_hot_target = torch.tensor(one_hot_target)
        
        loss = loss_function(y_pred, target.type(torch.long).to(device))
        
        metric = calculate_metrics(one_hot_target, y_pred, num_classes, metric)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        write_metrics(metric, num_classes, writer, step, loss)
        
        step += 1
    
    # Collect results for that epoch
    train_results = [{'accuracy': metric[i]['accuracy'],
                      'precision': metric[i]['precision'],
                      'recall': metric[i]['recall'],
                      'specificity': metric[i]['specificity']}
                     for i in range(num_classes)]

    return train_results, step


def calculate_metrics(true_labels, predictions, num_classes, metric):
    """
    Calculate tp, fp, fn, and tn of batch of samples.

    Parameters
    ----------
    true_labels : torch.Tensor
        Torch tensor containig the true labels.
    predictions : torch.Tensor
        Torch tensor containg the predicted labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    metric : list
        List of dictionaries containing tp, fp, fn, and tn for each class.

    """
    # Go through all classes and predictions
    for i in range(0, predictions.shape[0]):
        for j in range(0, num_classes):

            # If estimate and true label are greater than 0.5 then it is a tp
            if predictions[i][j] >= 0.5 and true_labels[i][j] >= 0.5:
                metric[j]['tp'] += 1

            # If estimate is less than 0.5 and true lable is greater than 0.5
            # then this is a fn
            elif predictions[i][j] < 0.5 and true_labels[i][j] >= 0.5:
                metric[j]['fn'] += 1

            # If estimate is greater than 0.5 and true label is less than 0.5
            # then this is a fp
            elif predictions[i][j] >= 0.5 and true_labels[i][j] < 0.5:
                metric[j]['fp'] += 1

            # If estimate and true label are less than 0.5 then it is a tn
            elif predictions[i][j] < 0.5 and true_labels[i][j] < 0.5:
                metric[j]['tn'] += 1

            tp = metric[j]['tp']
            fp = metric[j]['fp']
            fn = metric[j]['fn']
            tn = metric[j]['tn']

            # Calculate accuracy, precision, recall, and specificity
            if not (0 in [tp + tn + fp + fn]):
                metric[j]['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            if not (0 in [tp + fp]):
                metric[j]['precision'] = tp / (tp + fp)
            if not (0 in [tp + fn]):
                metric[j]['recall'] = tp / (tp + fn)
            if not (0 in [tn + fp]):
                metric[j]['specificity'] = tn / (tn + fp)

    return metric


def write_metrics(metric, num_classes, writer, step, loss=None, mode='Train',
                  include_count=False):
    """
    Write calculated metrics to tensorboard.

    Parameters
    ----------
    metric : list
        List of dict containing tp, fp, tn, and fn for each class.
    num_classes : int
        Number of classes in the output.
    writer : SummaryWriter
        Tensorboard SummaryWriter object.
    step : int
        The current step for the writer.
    loss : float, optional
        Loss calculatd between predictions and true labels.
        The default is 'None'.
    mode : str, optional
        Train, eval, or test. The default is 'Train'.
    include_count : bool, optional
        Boolean to include tp, fp, tn, and fn. The default is False.

    Returns
    -------
    None.

    """
    j = 0
    for i in range(0, num_classes):
        tp = metric[i]['tp']
        fp = metric[i]['fp']
        fn = metric[i]['fn']
        tn = metric[i]['tn']

        # Write accuracy, precision, recall, and specificity
        writer.add_scalar(f"{mode}_accuracy/{j}", metric[i]['accuracy'], step)
        writer.add_scalar(f"{mode}_precision/{j}", metric[i]['precision'], step)
        writer.add_scalar(f"{mode}_recall/{j}", metric[i]['recall'], step)
        writer.add_scalar(f"{mode}_specificity/{j}", metric[i]['specificity'], step)

        if include_count:
            writer.add_scalar(f"{mode}_true_negative/{j}", metric[i]['tn'], step)
            writer.add_scalar(f"{mode}_true_positive/{j}", metric[i]['tp'], step)
            writer.add_scalar(f"{mode}_false_negative/{j}", metric[i]['fn'], step)
            writer.add_scalar(f"{mode}_false_positive/{j}", metric[i]['fp'], step)
        
        j += 1

    if mode == "Train":
        writer.add_scalar('loss/train', loss, step)

    return None


def validation_step(model, valid_loader, device,
                    num_classes, epoch, writer, validation=True,
                    return_preds=False):
    """
    Perform the validation step in the epoch.

    Parameters
    ----------
    model : nn.Module
        Pytorch model.
    optimizer : torch.optim
        Optimizer used for training the model.
    loss_function : torch.loss
        Loss function used in the model.
    valid_loader : DataLoader
        DataLoader object containing the validation data.
    device : torch.device
        The device where the data is located.
    num_classes : int
        Number of outputs from the model.
    epoch : int
        Current elapsed epochs.
    writer : SummaryWriter
        Tensorboard SummaryWriter object.

    Returns
    -------
    valid_results : list
        List of dictionary containing accuracy, precision, recall,
        and specificity for each class.

    """
    # First set the model into evaluation mode
    model.eval()
    print("\nValidating\n")

    # Create metrics prior to starting validation loop
    metric = [{'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'accuracy': 0,
               'precision': 0, 'recall': 0, 'specificity': 0}
              for i in range(num_classes)]

    # Set to no grad
    with torch.no_grad():
        all_preds = None
        all_targets = None
        for sample in tqdm(valid_loader):
            sequence = sample['sequence']
            target = sample['label']
            if torch.cuda.is_available():
                model.cuda()
                sequence = sequence.cuda()
                target = target.cuda()
            y_pred = model(sequence)
            
            one_hot_target = []
            for i in range(sample['label'].shape[0]):
                one_hot_target.append(np.eye(num_classes)[sample['label'][i]])
        
            one_hot_target = torch.tensor(one_hot_target)

            # Calculate metrics
            metric = calculate_metrics(one_hot_target, y_pred, num_classes,
                                       metric)

            # Write metrics
            if validation:
                write_metrics(metric, num_classes, writer, step=epoch,
                              mode="Eval")
            
            if return_preds:
                if all_preds is None:
                    all_preds = y_pred.cpu()
                    all_targets = target.cpu()
                else:
                    all_preds = torch.cat((all_preds, y_pred.cpu()))
                    all_targets = torch.cat((all_targets, target.cpu()))
                    

    valid_results = [{'accuracy': metric[i]['accuracy'],
                      'precision': metric[i]['precision'],
                      'recall': metric[i]['recall'],
                      'specificity': metric[i]['specificity']}
                     for i in range(num_classes)]
    
    if return_preds:
        return valid_results, all_targets, all_preds
    
    return valid_results


def save_model(model, optimizer, TRAIN_PARAMS, epoch, log_dir):
    """
    Save the model in the specified directory as tar file.

    Parameters
    ----------
    model : nn.Module
        Pytorch model.
    optimizer : torch.optim
        Optimizer used for training.
    TRAIN_PARAMS : dict
        Parameters used for training.
    epoch : int
        Number of elapsed epochs.
    log_dir : str
        Location of the directory where file should be stored.

    Returns
    -------
    None.

    """
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training parameters': TRAIN_PARAMS},
               str(log_dir / (str(epoch).zfill(10) + ".tar")))

    return None


class StrokeDataset(Dataset):
    @timer
    def __init__(self, root_path, seq_len=120, drop_no_patient=False,
                 binary=False, viz='1D', rolling_window=False):
        self.metadata_path = root_path / 'metadata.csv'
        self.seq_len = seq_len
        self.metadata = pd.read_csv(self.metadata_path)
        # self.update_metadata()
        if drop_no_patient:
            self.remove_no_patient_data()
        if binary:
            self.drop_no_weakness_patients()
        if rolling_window:
            self.create_rolling_window()
        else:
            self.drop_smaller_sequences()
            self.fine_sequence_labels()
            self.remove_partial_sequences()
        self.transform = None
        self.viz = viz
        self.rw = rolling_window

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        samples : int
            Number of samples in the dataset.

        """
        # The length of the dataset is how many samples there are
        # In this case how many seq of data of specified length
        # First have to find all the coarse sequences
        # Next perform interger division using floor: len(seq) / seq_len
        # This gives the number of samples for one coarse seqeunce
        # Repeat with all the sequences
        # Sum the total and return the value
    
        # Each patient ID has coarse sequences
        # So have to iterate through each patient
        # Within each patient identify each seq len
        # For each seq len find how many samples
        if self.rw:
            return len(self.indexed_rw)        
        return len(self.metadata.groupby('Fine Seq ID'))

    # @timer
    # Get item function
    def __getitem__(self, idx):
        """
        Get a timeseries sample from the dataset.

        Parameters
        ----------
        idx : torch.Tensor, List
            The sequence index that needs to be acquired. This is the Fine Seq
            ID in the metadata.

        Returns
        -------
        sample : dict
            Dictionary of sequences of images and associated labels.

            sequence : torch.Tensor
                Continuous time series of pressure images. The length of the
                tensor is the specified seq_len.
            label : torch.Tensor
                Single label identifying the class of the sequence. There are
                three classes 0-left, 1-right, 2-no weakness.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.rw:
            # figure out how to get frames from self.metadata
            # create a dataframe of indexes within self.indexed_rw[idx]
            df = self.metadata.iloc[self.indexed_rw[idx]]

        else:    
            # idx is the fine sequence number
            indices = np.unique(self.metadata['Fine Seq ID'])
            df = self.metadata[self.metadata['Fine Seq ID'] == indices[idx]]
        
        seq_of_img = Parallel(n_jobs=1)(delayed(self.load_array)(df['Filename'], index) for index in range(len(df['Filename'])))
        
        seq_of_img = np.array(seq_of_img)
        
        # Reshape the array to have batch size first
        # seq_of_img = np.expand_dims(seq_of_img, axis=0)
        
        # Stack the array to mimic 3 channels 
        # seq_of_img = np.repeat(seq_of_img, 3, axis=0)
        
        # Get the labels
        label = df['Label']
        
        # Normalize the pressure values
        seq_of_img /= 2.66645
        
        # Create a sample
        sample = {'sequence': torch.from_numpy(seq_of_img),
                  'label': torch.from_numpy(label.to_numpy())[-1]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


    # Function to load in numpy array from metadata file
    def load_array(self, df, index):
        """
        Helper function to load a single image.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame object that holds the metadata.
        index : int
            The index of the dataframe from which the file should be loaded.

        Returns
        -------
        img : ndarray
            Flattened numpy array of the pressure image.

        """
        img_name = df.iloc[index]
        if self.viz == '1D':
            img = np.memmap(img_name, mode='r', dtype=np.float32, shape=(5664))
        else:
            img = np.memmap(img_name, mode='r', dtype=np.float32,
                            shape=(48, 118))
            # img = scipy.ndimage.zoom(img, 5)
        return img

    def fine_sequence_labels(self):
        """
        Assign label such that sequence is less than or equal to self.seq_len.

        The labels will be added to a new column in the metadata dataframe.
        Fine sequence labels will have metadata of pressure frames which belong
        to the same patient and are part of a continguous sequence of data.

        Returns
        -------
        None.

        """
        fine_seq_id = []
        counter = 0
        label = 0
        for row_name, dataframe in self.metadata.groupby(['Patient ID', 'Seq ID']):
            for _ in range(len(dataframe)):
                if (counter % self.seq_len == 0) and (counter != 0):
                    label += 1
                fine_seq_id.append(label)
                counter += 1
            counter = 0
            label += 1
        
        # Lastly add a column to the dataframe called Fine Seq ID
        self.metadata['Fine Seq ID'] = fine_seq_id

        return None

    # Create a main metadata file that combines all the metadata from the other
    # files
    # TODO: Fix this function so that it doesn't append to end of file
    @timer
    def update_metadata(self):
        if self.metadata_path.exists():
            # Read the csv
            self.metadata = pd.read_csv(self.metadata_path)
        else:
            # Create a new dataframe
            self.metadata = pd.DataFrame()
        # Next go through the directory structure and see if the metadata files are
        # part of the main meatadata file
        flist = [p for p in self.metadata_path.parent.glob("**/metadata.csv")]
        # Check if file in flist is subset and update if it is not
        # This only works if the file has some prior information
        self.check_subset(flist)
        # Save the updated metadata file
        self.metadata.to_csv(self.metadata_path, index=False)
    
    @timer
    def check_subset(self, flist):
        for file in flist:
            if file is not self.metadata_path:
                self.df = pd.read_csv(file)
                # The try is for the time if the metadata file is empty
                try:
                    _ = len(self.metadata.merge(self.df).drop_duplicates())
                except MergeError:
                    self.add_data(file)
                    self.df.to_csv(file, index=False)
                    self.metadata = pd.concat([self.metadata, self.df])
                if not (len(self.metadata.merge(self.df).drop_duplicates())
                        == len(self.metadata.drop_duplicates())):
                    # Do things to the dataframe if it is not a subset
                    # First append a col called patient_id
                    self.add_data(file)
                    self.df.to_csv(file, index=False)
                    frames = [self.metadata, self.df]
                    self.metadata = pd.concat(frames)
        return None

    @timer
    def find_sequences(self):
        """
        Find the number and length of continuos sequences in a dataframe.

        Returns
        -------
        seq_len : list
            List of sequences in the dataframe. The length of the list is the
            number of sequences and the index represents the number of frames
            in each sequence.

        """
        # Given a dataframe with the metadata of the pressure frames
        # Separate the frames out into continuos segments
        # There are time when the mattress is not recording because patient is
        # no longer on the mattress.
        seq_len = []
        for row in range(len(self.df['Datetime'])):
            self.df['Datetime'][row] = datetime.strptime(self.df['Datetime'][row],
                                                    "%Y-%m-%d %H:%M:%S")
        # Get a list of the timedeltas
        delta = (self.df['Datetime']-self.df['Datetime'].shift())
        # Find the values where delta is not 1 second
        num = 1  # Running counter; the first seq will have len=1
        for i in range(1, len(delta)):
            if delta[i].seconds >= 5:
                # Append to seq_len the number of frames in the seq
                seq_len.append(num)
                # Reset the number to 0
                num = 1
            else:
                num += 1
        seq_len.append(num)
        return seq_len

    # Function if the metadata file being created is blank
    # Have to concatenate the first metadata file and also add the Patient ID
    def add_data(self, file):
        """
        Add Patient ID and Seq ID information to metadata.

        Parameters
        ----------
        file : patlib.Path
            Path object for the location of the file.

        Returns
        -------
        None.

        """
        pat_id = [int(s) for s in re.findall(r'\d+', file.parent.stem)]
        self.df['Patient ID'] = pat_id[0]
        # Next find the number of continuos sequences
        sequences = self.find_sequences()
        # Append the sequence annotation to the dataframe
        seq_num = 0
        temp_arr = None
        for seq in sequences:
            if temp_arr is None:
                temp_arr = np.zeros(seq)
            else:
                temp_arr = np.concatenate((temp_arr,
                                          np.zeros(seq)+seq_num))
            seq_num += 1
        self.df['Seq ID'] = temp_arr
        return None
    
    @timer
    # Optional function to drop sequences that are smaller than self.seq_len
    def drop_smaller_sequences(self):
        # Group the sequences by pateint ID
        # Then go through each grouped dataframe
        # In each if the len of the seq_id is smaller than self.seq_len then
        # Drop the frame and save it to a new dataframe
        
        # First step; groupby and transform
        # This will only keep data that is longer than the seq_len
        self.metadata = self.metadata[self.metadata.groupby(['Patient ID', 'Seq ID'])['Frame'].transform('size') >= self.seq_len]
        

        return None
    
    @timer
    def remove_partial_sequences(self):
        self.metadata = self.metadata[self.metadata.groupby('Fine Seq ID')['Frame'].transform('size') >= self.seq_len]
        return None
    
    def remove_garbage_frames(self):
        # Have to open numpy files for this
        # Smarter way to do this is to open one sequence at a time
        # Get a dataframe for the first sequence
        # Open the first pressure frame of the data frame
        # Make the cutoff 0.07 (the standard for XSENSOR in psi)
        # The numpy array anything below 0.07 make it 0
        # Then see how many non-zero indices are there in the array
        # If the number of indices are above 0.9*5664 then it is garbage data
        # If the first data is garbage, then open the last pressure frame
        # See if it is also garbage
        # If the last frame is also garbage then the entire sequence is garbage
        # If the last frame is not garbage, go to the second frame and continue
        # continue until a non-grabage data is found
        # Drop all the frames that frames that are garbage data
        # Save the new metadata file in the drive
        seq = np.unique(self.metadata['Seq ID'])
        for s in seq:
            df = self.metadata[self.metadata['Seq ID'] == s]
            # Load the first file
            pres_arr = np.load(df['Filename'][0])
            
        pass
    
    @timer
    def remove_no_patient_data(self):
        self.metadata.drop(self.metadata[self.metadata['No Patient'] == True].index, inplace=True)
        # self.metadata.drop(['Garbage'], axis=1, inplace=True)
        self.metadata = self.metadata.astype({"Label": int})

    @timer
    def drop_no_weakness_patients(self):
        self.metadata.drop(self.metadata[self.metadata['Label'] == 2].index, inplace=True)

    def create_rolling_window(self):
        self.indexed_rw = rw(self.metadata, self.seq_len)
        
class XSNDataPreprocessor():
    """Class to preprocess an incoming pressure data file"""
    def __init__(self, df):
        self.df = df

    def get_patient_id(self):
        """
        

        Returns
        -------
        None.

        """
        assert len(self.df.columns) == 7, "Data is not raw, it has extra columns"
        # Get the patient id from the file name
        id_num = int(Path(self.df['Filename'].at[0]).parents[1].stem[1:4])
        # Add a column to df called Patient ID
        self.df['Patient ID'] = id_num
        return None

    def assign_seq_id(self, thresh=5):
        sequences = self.find_sequences()
        seq_num = 0
        temp_arr = None
        for seq in sequences:
            if temp_arr is None:
                temp_arr = np.zeros(seq)
            else:
                temp_arr = np.concatenate((temp_arr,
                                          np.zeros(seq)+seq_num))
            seq_num += 1
        self.df['Seq ID'] = temp_arr
        return None
        
    def find_sequences(self):
        """
        Find the number and length of continuos sequences in a dataframe.

        Returns
        -------
        seq_len : list
            List of sequences in the dataframe. The length of the list is the
            number of sequences and the index represents the number of frames
            in each sequence.

        """
        # Given a dataframe with the metadata of the pressure frames
        # Separate the frames out into continuos segments
        # There are time when the mattress is not recording because patient is
        # no longer on the mattress.
        seq_len = []
        for row in range(len(self.df['Datetime'])):
            self.df['Datetime'][row] = datetime.strptime(self.df['Datetime'][row],
                                                    "%Y-%m-%d %H:%M:%S")
        # Get a list of the timedeltas
        delta = (self.df['Datetime']-self.df['Datetime'].shift())
        # Find the values where delta is not 1 second
        num = 1  # Running counter; the first seq will have len=1
        for i in range(1, len(delta)):
            if delta[i].seconds >= 5:
                # Append to seq_len the number of frames in the seq
                seq_len.append(num)
                # Reset the number to 0
                num = 1
            else:
                num += 1
        seq_len.append(num)
        return seq_len

    def assign_label(self, label):
        self.df['Label'] = label
        return None

    @timer
    def add_metadata(self, label, thresh=5):
        self.get_patient_id()
        self.assign_seq_id(thresh)
        self.assign_label(label)
        return None

    def save_df(self, filename):
        self.df.to_csv(filename, index=False)
        return None

    @timer
    def add_no_patient_label(self, lower_thresh=0.2, upper_thresh=0.85):
        # Create a list for no patient labels
        no_pat_lab = []
        # List of for garbage data
        garbage_lab = []
        # Open each file individually
        for file in list(self.df['Filename']):
            arr = np.load(file)
            # Apply threshold of 0.07
            arr[arr < 0.07] = 0
            if np.count_nonzero(arr) <= (arr.shape[0] * arr.shape[1] * lower_thresh):
                no_pat_lab.append(True)
            else:
                no_pat_lab.append(False)
            if np.count_nonzero(arr) >= (arr.shape[0] * arr.shape[1] * upper_thresh):
                garbage_lab.append(True)
            else:
                garbage_lab.append(False)
        self.df['No Patient'] = no_pat_lab
        self.df['Garbage'] = garbage_lab
        return None

    @timer
    def add_no_patient_label_parallel(self, lower_thresh=0.2, upper_thresh=0.85):
        # Create a list for no patient labels
        self.no_pat_lab = []
        # List of for garbage data
        self.garbage_lab = []
        no_pat_lab = Parallel(n_jobs=1)(delayed(self.check_garbage_or_no_patient)(file, lower_thresh, upper_thresh) for file in list(self.df['Filename']))
        self.df['No Patient'] = no_pat_lab
        # self.df['Garbage'] = garbage_lab
        return None
    
    def check_garbage_or_no_patient(self, file, low=0.2, high=0.85):
        arr = np.load(file)
        arr[arr < 0.07] = 0
        if np.count_nonzero(arr) <= (arr.shape[0] * arr.shape[1] * low):
            no_pat_lab = True
        else:
            no_pat_lab = False
        if np.count_nonzero(arr) >= (arr.shape[0] * arr.shape[1] * high):
            garbage_lab = True
        else:
            garbage_lab = False
        return no_pat_lab

@timer
# Function to time how long it takes to pull out a sample from dataloader
def time_dataloader(dataset):
    # Initialise dataset
    sample_loader =  DataLoader(dataset, batch_size=2, num_workers=16)
    i = 0
    for sample in sample_loader:
        assign_sequence(sample)
        i += 1
        if i > 1:
            break
    return None


@timer
# Function to setup the dataloader
def setup_dataloader(dataset):
    sample_loader =  DataLoader(dataset, batch_size=2, num_workers=1)
    return sample_loader


@timer
# Function to get next item in dataloader
def get_next_item(dataloader):
    return next(dataloader)


@timer
# Function to call in one sample at a time
def time_dataloader_v2(dataset):
    sample_loader = setup_dataloader(dataset)
    foo = make_iterable(sample_loader)
    for _ in range(100):
        get_next_item(foo)
    return None


@timer
# Function to assign sequence
def assign_sequence(sample):
    sequence = sample['sequence']
    return sequence

@timer
# make a dataloader into an iterable
def make_iterable(dataloader):
    return iter(dataloader)



class IterableStrokeDataset(Dataset):
    
    def __init__(self,root_path, seq_len=120, drop_no_patient=False,
                 binary=False, viz='1D'):
        pass


if __name__ == '__main__':
    # # torch.backends.cudnn.enabled = False
    binary = True
    root_path = Path(r'D:\Aakash\Masters\Binary\Train')
    datset = StrokeDataset(root_path, seq_len=360, binary=binary,
                            drop_no_patient=True, viz='1D')
    foo = datset.__getitem__(1)
    
    valid_root_path = Path(r'D:\Aakash\Masters\Binary\Validation')
    valid_dataset = StrokeDataset(valid_root_path, seq_len=360, binary=binary,
                                  drop_no_patient=True, viz='1D', rolling_window=True)
    
    # # Code for testing how long it takes to load one sample
    # # root_path = Path(r'C:\Users\BTLab\Documents\Aakash\Patient Data from Stroke Ward\p002')
    # # dataset = StrokeDataset(root_path, seq_len=128)
    # # time_dataloader_v2(dataset)
    
    # model = StrokeNet(num_conv_layers=0, input_dim=5664, seq_len=360, pool='fc', binary=binary)
    # # model = StrokeInceptionNet()
    # model.load_state_dict(torch.load(r"E:\Models\dropout_0.7_6min_every_epoch\0000000047.tar")['model'])
    # # model= StrokeNetV2(256)
    # if torch.cuda.is_available():
    #     model.cuda()
    # log_dir = Path(r'E:\Models\dropout_0.7_6min_every_epoch')
    # train_scores, val_scores, model = train_model(model, datset, log_dir,
    #                                               k_fold=0, binary=binary,
    #                                               validation_dataset=valid_dataset,
    #                                               save_every_epoch=True)
    
    # # Code to test the model
    # test_root_path = Path(r'D:\Aakash\Masters\Stroke Patient Metadata')
    # list_of_patient_paths = [Path(f.path) for f in os.scandir(test_root_path) if (f.is_dir() and Path(f.path).stem != 'p012' and Path(f.path).stem != 'p003' and Path(f.path).stem != 'p008' and Path(f.path).stem != 'p015')]
    
    # patient_test_scores = {}
    # patient_preds = {}
    # patient_targets = {}
    # for patient in list_of_patient_paths:
    #     test_dataset = StrokeDataset(patient, seq_len=360, binary=binary, drop_no_patient=True, viz='1D')
    #     test_results, y_true, y_pred = test_model(model, test_dataset, log_dir,
    #                                               force_binary=binary)
    #     patient_test_scores[patient.stem] = test_results
    #     patient_preds[patient.stem] = y_pred
    #     patient_targets[patient.stem] = y_true
        
    
    # fig, ax = plt.subplots(figsize=(16, 12))
    # plot_roc(y_true, y_pred, title='ROC Curves for testing data in the LSTM model with step decaying learning rate scheduler 4min long sequences, with low initial learning rate', ax=ax)
    
    #        10,10,11,12,13,14,14,15,16,17,18,18,19,20,21,22,23,24,25,26
    # labels = [2, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1]
    # i = 0
    
    # # Code for getting the new data prepped
    # for fn in list(Path(r'D:\Aakash\Masters\Stroke Patient Data').glob('**/*.csv')):
    #     df = pd.read_csv(fn)
    #     df = XSNDataPreprocessor(df)
    #     df.add_metadata(labels[i])
    #     df.add_no_patient_label_parallel()
    #     df.save_df(fn)
    #     i += 1
