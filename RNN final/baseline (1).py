import time
from torch import Tensor
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import pandas as pd
#for rmse calculator
import math
from sklearn.metrics import mean_squared_error



#decided to just use mse_loss
# class RMSE(nn.Module):
#     def __init__(self, epsilon=1e-6):
#         super(RMSE, self).__init__()
#         # eps in case mse = 0 which produces NaN value
#         self.epsilon = epsilon

#     def forward(self, y_pred, y_true):
#         # rmse is square root
#         rmse_loss = torch.sqrt(F.mse_loss(y_pred, y_true) + self.epsilon)
#         return rmse_loss


class Baseline(nn.Module):
    def training_step(self, x_b, y_b):
        pred = self(x_b)
        loss = F.mse_loss(pred, y_b)
        return loss

    def validation_step(self, x_v, y_v, test: bool = False) -> dict:
        #grab prediction and calculate loss return as dict
        v_pred = self(x_v)
        loss = F.mse_loss(v_pred, y_v)
        if test:
            return v_pred, {'test_loss': loss.detach()}
        else:
            return {'val_loss': loss.detach()}

    def validation_epoch_end(self,
                             outputs: dict,
                             test: bool = False) -> dict:
        batch_losses = [x['val_loss'] for x in outputs]
        # stack to combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        # print every five epochs
        if (epoch) % 5 == 0:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss']))


def evaluate(model: nn.Module,
             test_loader: torch.utils.data.DataLoader,
             batch_size: int = 1,
             num_feat: int = 1):
    with torch.no_grad():
        device = get_default_device()
        values = []
        predictions = []
        best_test_loss = float('inf')
        for x_t, y_t in test_loader:
            #reshape x
            x_t = x_t.view([batch_size, -1, num_feat]).to(device)
            y_t = y_t.to(device)
            model.eval()
            v_pred, loss = model.validation_step(x_t, y_t, test=True)
            #find best rmse score. 
            if loss['test_loss'] < best_test_loss:
                best_test_loss = loss['test_loss']
            predictions.append(v_pred.detach().cpu().numpy())
            values.append(y_t.detach().cpu().numpy())
        return predictions, values, best_test_loss


def fit(model: torch.nn.Module,
        train_loader,
        val_loader,
        epochs: int = 50,
        lr: float = 0.001,
        opt_func: str = 'SGD',
        batch_size: int = 64,
        num_feat: int = 1,
        ROPlateau: bool = False):

    torch.cuda.empty_cache()
    device = get_default_device()
    begin = time.time()
    history = []

    # organise optimisers
    if opt_func == 'SGD':
        optimiser = get_optimiser(opt_func)(
            model.parameters(), lr=lr, momentum=0.9)
    else:
        optimiser = get_optimiser(opt_func)(
            model.parameters(), lr=lr)

    if ROPlateau == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min",
            patience=0,
            cooldown=2,
            verbose=False, factor=0.5,
            min_lr=1e-6)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        batch_loss = []
        for i, (x_b, y_b) in enumerate(train_loader):
            x_b = x_b.view([batch_size, -1, num_feat]).to(device)
            y_b = y_b.to(device)
            # enter training
            model.train()
            loss = model.training_step(x_b, y_b)
            batch_loss.append(loss)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        # move to validation set
        with torch.no_grad():
            validation_b_loss = []
            for i, (x_v, y_v) in enumerate(val_loader):
                x_v = x_v.view([batch_size, -1, num_feat]).to(device)
                y_v = y_v.to(device)
                model.eval()
                v_loss = model.validation_step(x_v, y_v)
                # monitor best val_loss
                if v_loss['val_loss'] < best_val_loss:
                    best_val_loss = v_loss['val_loss']
                    model_best_model_state_dict = model.state_dict()

                validation_b_loss.append(v_loss)
            total_loss = model.validation_epoch_end(validation_b_loss)
            # add train_loss to dictionary
            total_loss['train_loss'] = torch.stack(batch_loss).mean().item()
            model.epoch_end(epoch, total_loss)
            history.append(total_loss)
            # save models for testing
    model.best_model_path = f'model:{model.model_name}, Run:{model.model_run_no}.pt'
    torch.save(model_best_model_state_dict, model.best_model_path)
    finish = time.time()
    print(f'Training Duration: {(finish-begin)/60:.2f} minutes.')
    return history


# Function to get appropriate optimiser
def get_optimiser(optimiser: str = "Adam") -> torch.optim:
    if optimiser == "Adam":
        optimiser = torch.optim.Adam
    elif optimiser == "AdamW":
        optimiser = torch.optim.AdamW
    elif optimiser == "Adamax":
        optimiser = torch.optim.Adamax
    elif optimiser == "SGD":
        optimiser = torch.optim.SGD
    elif optimiser == "RMSprop":
        optimiser = torch.optim.RMSprop
    elif optimiser == "Nadam":
        optimiser = torch.optim.NAdam
    else:
        raise Exception(
            "Unknown optimiser. Adam/AdamW/Adamax/SGD/SparseAdam/Adadelta")
    return optimiser



def result(model,
            scaler,
            sequence,
            model_name: str,
            model_run_no: int,
            test_loader: torch.utils.data.dataloader.DataLoader,
            df: pd.DataFrame,
            x_train: np.ndarray,
            batch_size: int = 32,
            num_feat: int = 1,
            diff_testset: bool = False,
            ):
    #load model path
    model_path = f"/content/model:{model_name}, Run:{model_run_no}.pt"
    model.load_state_dict(torch.load(model_path))
    #evaluate on testing
    predictions, values, loss = evaluate(
        model=model, test_loader=test_loader, batch_size=batch_size, num_feat=num_feat)
    rescaled_data, RMSE = inverse_transformer(
        df, scaler, predictions, values, x_train, diff_testset, sequence)
    # return rescaled_data, loss
    return rescaled_data, RMSE

def inverse_transformer(df, scaler, predictions, values, x_train, diff_testset, sequence):
    prediction = np.stack(predictions)
    value = np.stack(values)
    # pad left of target with zeros to make dims = orinal transform dims
    predictions1 = prediction.reshape(-1, 1)
    values1 = value.reshape(-1, 1)
    #if scaling occured with target in two columns
    if diff_testset:
        #exact same but with train_shape kept the same
        preds = np.zeros(shape=(len(predictions1), x_train.shape[2]))
        preds[:, -1] = predictions1[:, 0]
        # transform final column all rows
        preds = scaler.inverse_transform(preds)[:, -1]
        #same for ground truth
        vals = np.zeros(shape=(len(values1), x_train.shape[2]))
        vals[:, -1] = values1[:, 0]
        vals = scaler.inverse_transform(vals)[:, -1]
        #calculate RMSE on rescaled values
        RMSE = math.sqrt(mean_squared_error(preds.reshape(-1,1), vals.reshape(-1,1)))
        length = len(preds)
        pred_index = df.iloc[-length:, :]
        df_result = pd.DataFrame(
            data={"value": vals, "prediction": preds}, index=pred_index.index)
        print(f'Test RMSE: {round(RMSE,3)}')
        return df_result, RMSE
    else:
        preds = np.zeros(shape=(len(predictions1), x_train.shape[2]+1))
        preds[:, -1] = predictions1[:, 0]
        # transform final column all rows
        preds = scaler.inverse_transform(preds)[:, -1]
        #same for ground truth
        vals = np.zeros(shape=(len(values1), x_train.shape[2]+1))
        vals[:, -1] = values1[:, 0]
        vals = scaler.inverse_transform(vals)[:, -1]
        #calculate RMSE on rescaled values
        RMSE = math.sqrt(mean_squared_error(preds, vals))
        pred_index = df.iloc[-len(preds):, :]
        df_result = pd.DataFrame(
            data={"value": vals, "prediction": preds}, index=pred_index.index)
        print(f'Test RMSE: {round(RMSE,3)}')
        return df_result, RMSE
