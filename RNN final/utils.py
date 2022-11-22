from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''

I wrote a lot of added functionality for a whole range of tests.
However I quickly realised when running the experiments that
I wouldnt able to use very much of it in order to save space on the notebook

'''



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return None


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


# merge stocks
def stock_merger(stock1: str,
                 stock2: str,
                 stock3: str):

    df1 = pd.read_csv(f'stocks/{stock1}.csv', index_col=0, parse_dates=True)
    df2 = pd.read_csv(f'stocks/{stock2}.csv', index_col=0, parse_dates=True)
    df3 = pd.read_csv(f'stocks/{stock3}.csv', index_col=0, parse_dates=True)

    df1 = df1.rename(columns={'Close': f'{stock1}_close'})
    df2 = df2.rename(columns={'Close': f'{stock2}_close'})
    merged = pd.concat([df1, df2, df3], axis=1)
    merged = merged.dropna(axis=0)
    return merged


#helper function for data_sequence function
def sequencer(array, sequence):
    # sequences to be used
    sequenced_data = [array[i: i + sequence]
                      for i in range(len(array) - sequence)]
    sequenced_data = np.array(sequenced_data)

    return sequenced_data

# scaling function for data_sequence function

def split_scale(val_split: float = .2,
                test_split: float = .2,
                n_data: list = [],
                n_test: list = [],
                diff_testset: bool = False):
    if diff_testset:
        # organise splits and scale
        scaler = MinMaxScaler()
        validation_len = int(np.round(val_split*n_data.shape[0])) 
        train_len = n_data.shape[0] - validation_len
        train = n_data[:train_len]
        val = n_data[train_len: train_len + validation_len]
        train_scale = scaler.fit_transform(train)
        #only transform val and test to not leak
        val_scale = scaler.transform(val)
        #pad test set to have same size 
        test_set = np.zeros(shape=(len(n_test), train.shape[1]))
        n_test = n_test.reshape(-1,1)
        test_set[:, -1] = n_test[:, 0]
        # transform final column all rows
        test_scale = scaler.transform(test_set)[:, -1].reshape(-1,1)
        return train_scale, val_scale, test_scale, scaler
    else:
        # test set comes from end of data
        scaler = MinMaxScaler()
        validation_len = int(np.round(val_split*n_data.shape[0]))
        test_len = int(np.round(test_split*n_data.shape[0]))
        train_len = n_data.shape[0] - (validation_len + test_len)
        train = n_data[:train_len]
        val = n_data[train_len: train_len + validation_len]
        test = n_data[train_len + validation_len:]
        train_scale = scaler.fit_transform(train)
        # only transform val and test to not leak
        val_scale = scaler.transform(val)
        test_scale = scaler.transform(test)
        return train_scale, val_scale, test_scale, scaler


# this is a big function to organise training splits/normalisation/batchloaders
# the separate test functionality was used to match arima test set exactly
def data_sequence(dataframe: pd.DataFrame,
                  target: str = 'Close',
                  date_start: str = 'Empty',
                  date_end: str = 'Empty',
                  pred_date_start: str = '',
                  pred_date_end: str = '',
                  features: list = [],
                  sequence: int = 10,
                  batch_size: int = 32,
                  include_target: bool = True,
                  diff_testset=False
                  ):

    # condition for using whole dataset or slice
    if date_start == 'Empty' and date_end == 'Empty':
        
        dataframe1 = dataframe.copy()

        # Condition to make test set a separate year
        if diff_testset == False:
            # if not we need our target at end of df
            t = dataframe1.pop(target)
            dataframe1 = dataframe1.drop(columns=[i for i in features], axis=1)
            dataframe1 = dataframe1.assign(target=t)

            # if we include target in training need to add it again
            if include_target == True:
                dataframe1.insert(0, target, t)
            n_test = []
            n_data = dataframe1.values
        else:
            # organise separate test dates
            test_set = dataframe1.loc[pred_date_start:pred_date_end]
            test_set = test_set[target]
            training_set = dataframe1.drop(columns=[i for i in features], axis=1)
            training_set = dataframe1.loc[:pred_date_start]
            n_test = test_set.values
            n_data = training_set.values
    # do the same for sliced dataset
    else:
        dataframe1 = dataframe.copy()
        if diff_testset == False:
            t = dataframe1.pop(target)
            dataframe1 = dataframe1.drop(columns=[i for i in features], axis=1)
            dataframe1 = dataframe1.assign(target=t)
            if include_target == True:
                dataframe1.insert(0, target, t)
            # slice dataset
            dataframe1 = dataframe1.loc[date_start:date_end]
            n_test = []
            n_data = dataframe1.values
        else:
            #organise separate testset
            test_set = dataframe1.loc[pred_date_start:pred_date_end]
            test_set_series = test_set[target]
            training_set = dataframe1.drop(columns=[i for i in features], axis=1)
            training_set = training_set.loc[date_start:date_end]
            n_test = test_set_series.values
            n_data = training_set.values


    #scale data and organise into correct sequences
    train, val, test, scaler = split_scale(.2, .2, n_data, n_test, diff_testset)
    s_tr = sequencer(train, sequence)
    s_v = sequencer(val, sequence)
    s_t = sequencer(test, sequence)

    # return train, val, test
    if diff_testset:
      x_train = s_tr[:, :-1, :]
      x_validation = s_v[:, :-1, :]
      x_test = s_t[:, :-1, :]
      
    elif include_target:
        # separate datasets so no double of target column
        feats = n_data.shape[1]
        x_train = s_tr[:, :-1, :feats-1]
        x_validation = s_v[:, :-1, :feats-1]
        x_test = s_t[:, :-1, :feats-1]
    else:
        #double functionality not needed in the end
        #will leave in case I return to orginal
        feats = n_data.shape[1]
        x_train = s_tr[:, :-1, :feats-1]
        x_validation = s_v[:, :-1, :feats-1]
        x_test = s_t[:, :-1, :feats-1]
    
    y_train = s_tr[:, -1, -1:]
    y_validation = s_v[:, -1, -1:]
    y_test = s_t[:, -1, -1:]
    print(f'X_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'X_validation.shape = {x_validation.shape}')
    print(f'y_validation.shape = {y_validation.shape}')
    print(f'X_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    # organise data into tensors and dataloaders
    train_features = torch.Tensor(x_train)
    train_targets = torch.Tensor(y_train)
    val_features = torch.Tensor(x_validation)
    val_targets = torch.Tensor(y_validation)
    test_features = torch.Tensor(x_test)
    test_targets = torch.Tensor(y_test)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, drop_last=True)
    #test loader with one in case I want to make preds one at a time
    pred_one_test_loader = DataLoader(
        test, batch_size=1, shuffle=False, drop_last=True)

    if diff_testset:
        return [train_loader, val_loader, test_loader, pred_one_test_loader, scaler, test_set, x_train, sequence]
    else:
        return [train_loader, val_loader, test_loader, pred_one_test_loader, scaler, dataframe1, x_train, sequence]


def loss_curves(history: list,
                caption: str = "Loss Curves",
                experiment_no: int = 1,
                lr: float = 0.001,
                epochs: int = 30,
                network: str = ""
                ) -> None:
    # arrays of losses
    train = [instance['train_loss'] for instance in history]
    val = [instance['val_loss'] for instance in history]

    plt.style.use('seaborn')
    plt.figure(figsize=(8, 6))
    plt.plot(train, label="Training Loss", c="r")
    plt.plot(val, label="Validation Loss", c="b")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Experiment no. {experiment_no}, {caption}: lr={lr}\nEpochs={epochs}, network type={network}")
    plt.legend()
    plt.show()

# visualisation
def plot_results(df, stock, target, model, sequence):
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 8))
    plt.plot(df['prediction'], label="Predicted", c="orange")
    plt.plot(df['value'], label="Ground Truth", c="g")
    plt.xlabel("Date", fontsize=20)
    plt.ylabel(target, fontsize=20)
    plt.title(f'{model}: Sequence - {sequence}', fontsize=25)
    plt.legend()
    plt.show()
