from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
MAX_ATOMIC_NUM = 100



def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(self.hparams.encoder)

        self.task = self.hparams.task

    def nanmean(self, input_tensor):
        mask = torch.isnan(input_tensor)
        input_tensor = torch.where(mask, torch.zeros_like(input_tensor), input_tensor)
        sum_tensor = torch.sum(input_tensor)
        count_tensor = torch.sum(~mask).type(input_tensor.dtype)
        nanmean_tensor = sum_tensor / count_tensor

        return nanmean_tensor

    def pearson_correlation(self, preds, labels):

        # Compute the mean of predictions and labels
        preds_mean = torch.mean(preds)
        labels_mean = torch.mean(labels)

        # Compute the deviations from the mean
        preds_dev = preds - preds_mean
        labels_dev = labels - labels_mean

        # Compute the covariance
        covariance = torch.sum(preds_dev * labels_dev)

        # Compute the standard deviations
        preds_std = torch.sqrt(torch.sum(preds_dev**2))
        labels_std = torch.sqrt(torch.sum(labels_dev**2))

        # Compute the Pearson correlation coefficient
        pearson_corr = covariance / (preds_std * labels_std)

        return pearson_corr

    def get_loss(self, preds, labels):

        if self.task == 'regression':

            loss = F.huber_loss(preds, labels, reduction='none')
        
        elif self.task == 'classification':
            loss = F.cross_entropy(preds, labels.reshape(-1), reduction='none')

        return loss

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        preds = self.encoder(batch)  # shape (N, 1)
        return preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        loss = self.get_loss(preds, batch.y)

        loss = loss.mean()



        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )


        if torch.isnan(loss):
            loss = None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        preds = self(batch)

        log_dict, loss = self.compute_stats(batch, preds, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, preds, prefix):
        loss = self.get_loss(preds, batch.y)
        loss = self.nanmean(loss)

        self.scaler.match_device(preds)

        if self.task == 'regression':
            
            pcc = self.pearson_correlation(preds.reshape(-1), batch.y.reshape(-1))

            log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_pcc': pcc,
            }
        else:
            dis_preds = preds.argmax(dim=-1)
            dis_y = batch.y.reshape(-1)
            acc = torch.mean(dis_preds == dis_y)
            log_dict = {
                f'{prefix}_loss': loss,
                f'{prefix}_acc': acc,
            }        

        return log_dict, loss



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )
    return model


if __name__ == "__main__":
    main()
