from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline import Baseline 
from utils import get_default_device
#for arima
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


#these models were inspired by this article https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b


class LSTM(Baseline):
    def __init__(self,
                 in_dims: int = 1,
                 hid_dims: int = 3,
                 no_layers: int = 2,
                 out_dim: int = 1,
                 model_name: str = 'LSTM',
                 model_run_no: int = 1,
                 drop_prob: float = .5):
        super(LSTM, self).__init__()
        self.hid_dims = hid_dims
        self.no_layers = no_layers
        self.model_name = model_name
        self.model_run_no = model_run_no

        self.lstm = nn.LSTM(in_dims, hid_dims, no_layers,
                            dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hid_dims, out_dim)
  
    def forward(self, x):
        device = get_default_device()
        # keep track of two states: hidden and cell
        hidden_state = torch.zeros(self.no_layers, x.size(
            0), self.hid_dims).requires_grad_().to(device)
        cell_memory = torch.zeros(self.no_layers, x.size(
            0), self.hid_dims).requires_grad_().to(device)
        out, (next_hs, next_cn) = self.lstm(
            x, (hidden_state.detach(), cell_memory.detach()))
        out = self.fc(out[:, -1, :])
        return out


# this model is essentially the same as the LSTM code minus the Cell State
class GRU(Baseline):
    def __init__(self,
                 in_dims: int = 1,
                 hid_dims: int = 3,
                 no_layers: int = 2,
                 out_dim: int = 1,
                 model_name: str = 'GRU',
                 model_run_no: int = 1,
                 drop_prob: float = .5):
        super(GRU, self).__init__()
        self.hid_dims = hid_dims
        self.no_layers = no_layers
        self.model_name = model_name
        self.model_run_no = model_run_no

        self.gru = nn.GRU(in_dims, hid_dims, no_layers,
                          dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hid_dims, out_dim)

    def forward(self, x: Tensor):
        device = get_default_device()
        # only hidden state
        hidden_state = torch.zeros(self.no_layers, x.size(
            0), self.hid_dims).requires_grad_().to(device)
        #truncated backprop so detach
        out, _ = self.gru(x, (hidden_state.detach()))
        out = self.fc(out[:, -1, :])
        return out

#similar to the gru code but need to pass hidden state back to model
class RNN(Baseline):
    def __init__(self,
                 in_dims: int = 1,
                 hid_dims: int = 3,
                 no_layers: int = 2,
                 out_dim: int = 1,
                 model_name: str = 'RNN',
                 model_run_no: int = 1,
                 drop_prob: int = .5):
        super(RNN, self).__init__()
        self.hid_dims = hid_dims
        self.no_layers = no_layers
        self.model_name = model_name
        self.model_run_no = model_run_no

        self.rnn = nn.RNN(in_dims, hid_dims, no_layers,
                          dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hid_dims, out_dim)

    def forward(self, x: Tensor):
        device = get_default_device()
        #init hidden state
        hidden_state = torch.zeros(self.no_layers, x.size(
            0), self.hid_dims).requires_grad_().to(device)
        #pass input and new hidden state in
        out, hidden_state = self.rnn(x, hidden_state.detach())
        out = self.fc(out[:, -1,:])
        return out




def arima(train: np.ndarray, 
          test: np.ndarray,
          df_test: pd.DataFrame,
          order: tuple = (5,1,0),
          stock: str = 'ABC',
          target: str= 'Close',
          ):
  
        x1 = train.reshape(-1,1)
        x2 = test.reshape(-1,1)

        #scale data
        scaler = MinMaxScaler()
        train = scaler.fit_transform(x1)
        test = scaler.fit_transform(x2)

        train = list(train)
        test = list(test)

        model_preds = []
        steps = len(list(test))
        for i in range(steps):
          model = ARIMA(train, order = order)
          model_fit = model.fit()
          #forecast
          output = model_fit.forecast()
          y_pred = output[0]
          model_preds.append(y_pred)
          ground_truth = test[i]
          #update trainset with previous gt
          train.append(ground_truth)
        
        #transform back and plot results
        preds = np.stack(model_preds).reshape(-1,1)
        vals = np.stack(test).reshape(-1,1)
        predictions = scaler.inverse_transform(preds)
        values = scaler.inverse_transform(vals)
        RMSE = np.sqrt(mean_squared_error(predictions, values))
        plt.figure(figsize=(15,9))
        plt.grid(True)
        plt.title('Ground Truth vs Predictions')
        plt.xlabel('Dates')
        plt.ylabel(f'{stock} {target}')
        plt.plot(df_test.index, predictions, 'orange', marker='o', linestyle='dashed', label='Pred')
        plt.plot(df_test.index, values, 'green',label='Ground Truth')
        plt.legend()
        plt.show()
        print(f'Testing RMSE: {round(RMSE, 3)}')