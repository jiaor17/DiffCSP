import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from diffcsp.pl_modules.embeddings import MAX_ATOMIC_NUM, MAX_SPACE_GROUP_NUM
from diffcsp.pl_modules.embeddings import KHOT_EMBEDDINGS



class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

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




def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class EncTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ff_dim,
        num_layers,
        head,
        dropout,
        leaky_relu_slope,
        dense_output_nonlinearity,
        return_node = False
    ):
        super(EncTransformer, self).__init__()
        c = copy.deepcopy
        # d_atom, d_space_group, d_pos, N=2, d_model=128, d_ff=512, h=8, dropout=0.1, 
        #        leaky_relu_slope=0.0, 
        #        dense_output_nonlinearity='relu', 
        #        scale_norm=False,  d_proj_list=None
        attn = MultiHeadedAttention(head, hidden_dim, dropout)
        ff = PositionwiseFeedForward(hidden_dim, ff_dim, dropout, leaky_relu_slope, dense_output_nonlinearity)
        self.encoder = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), num_layers)
        self.src_embed = nn.Embedding(MAX_ATOMIC_NUM + 1, hidden_dim)
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.return_node = return_node
        # self.feat_embed = nn.Embedding(MAX_SPACE_GROUP_NUM, hidden_dim)

    @property
    def device(self):
        return self.dummy_param.device

    def get_edges(self, batch):

        batch_size = batch.num_graphs
        # [B, N]
        lis = [torch.ones(n,n, device=self.device) for n in batch.num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        fc_edges = fc_edges + batch_size
        vn_i = torch.repeat_interleave(torch.arange(batch_size, device=self.device), batch.num_atoms)
        vn_j = torch.arange(batch.num_nodes, device=self.device) + batch_size
        vn = torch.cat([vn_i.unsqueeze(0), vn_j.unsqueeze(0)])
        vn_r = torch.cat([vn_j.unsqueeze(0), vn_i.unsqueeze(0)])
        vn_self = torch.arange(batch_size, device=self.device)
        vn_loop = torch.cat([vn_self.unsqueeze(0), vn_self.unsqueeze(0)])
        edges = torch.cat([fc_edges, vn, vn_r, vn_loop], dim=-1)

        return edges
        
    # def forward(self, src, src_mask, feat, pos):
    def forward(self, batch):
        "Take in and process masked src and target sequences."

        edges = self.get_edges(batch)

        dummy_tokens = torch.zeros(batch.num_graphs, device=self.device).long()
        embed = torch.cat((self.src_embed(dummy_tokens), self.src_embed(batch.atom_types)),dim=0)
        enc = self.encoder(embed, edges)
        if self.return_node:
            return enc[:batch.num_graphs], enc[batch.num_graphs:]
        return enc[:batch.num_graphs]
    
class PredTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ff_dim,
        num_layers,
        head,
        dropout,
        leaky_relu_slope,
        dense_output_nonlinearity
    ):
        super(PredTransformer, self).__init__()
        self.enc = EncTransformer(hidden_dim, ff_dim, num_layers, head, dropout, leaky_relu_slope, dense_output_nonlinearity)
        self.fc_property = build_mlp(hidden_dim, hidden_dim, 2, 1)

    def forward(self,batch):
        
        enc_ = self.enc(batch)
        prop = self.fc_property(enc_)
        return prop

class Generator(nn.Module):
    def __init__(self, d_model,  
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False, d_proj_list=None):
        super(Generator, self).__init__()
        self.lattice_gen = LatticeGenerator(d_model, d_proj_list, leaky_relu_slope, dropout)
        self.pos_gen = PositionGenerator(d_model, d_proj_list, leaky_relu_slope, dropout)

    def forward(self, x, batch_size):
        pos = self.pos_gen(x[batch_size:])
        lat = self.lattice_gen(x[:batch_size])
        return pos, lat
    
class PositionGenerator(nn.Module):
    def __init__(self, d_model, d_proj_list=None, leaky_relu_slope=0.01, dropout=0.0):
        super(PositionGenerator, self).__init__()
        # self.norm = LayerNorm(d_model)
        self.d_model = d_model
        if d_proj_list is None:
            self.proj = nn.Linear(d_model, 3)
        else:
            self.proj = []
            for din, dout in zip([self.d_model] + d_proj_list[:-1], d_proj_list):
                self.proj.append(nn.Linear(din, dout))
                self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_proj_list[-1], 3))
            self.proj = torch.nn.Sequential(*self.proj)

    def forward(self, x):
        projected = self.proj(x)
        return projected

class LatticeGenerator(nn.Module):
    def __init__(self, d_model, d_proj_list=None, leaky_relu_slope=0.01, dropout=0.0):
        super(LatticeGenerator, self).__init__()
        self.d_model = d_model
        if d_proj_list is None:
            self.proj = nn.Linear(d_model, 6)
        else:
            self.proj = []
            for din, dout in zip([self.d_model] + d_proj_list[:-1], d_proj_list):
                self.proj.append(nn.Linear(din, dout))
                self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_proj_list[-1], 6))
            self.proj = torch.nn.Sequential(*self.proj)
    def forward(self, x):
        projected = self.proj(x)
        return projected
    

### Encoder

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, edges):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, edges))
        return self.sublayer[1](x, self.feed_forward)
    

def attention(query, key, value, edges, dropout=None, 
              eps=1e-6, inf=1e12):
    "Compute 'Scaled Dot Product Attention'"
    # q,k,v N * h * d_k
    d_k = query.size(-1)
    query_i, key_j = query[edges[0]], key[edges[1]] # E * h * d_k
    scores = (query_i * key_j).sum(dim=-1) / math.sqrt(d_k) # E * h
    p_attn = scatter_softmax(scores, edges[0], dim=0) # E * h
    
    if dropout is not None:
        p_attn = dropout(p_attn)

    atoms_features = p_attn.unsqueeze(-1) * value[edges[1]] # E * h * d_k
    atoms_features = scatter(atoms_features, edges[0], dim=0, reduce='sum')
    return atoms_features, p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
            
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, edges):
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(-1, self.h, self.d_k) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.self_attn = attention(query, key, value, edges, dropout=self.dropout) 
        # 3) "Concat" using a view and apply a final linear. 
        x = x.view(-1, self.h * self.d_k)
        return self.linears[-1](x)


### Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.dropout = clones(nn.Dropout(dropout), 2)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
            

    def forward(self, x):
        
        x = self.dropout[0](F.leaky_relu(self.ff_in(x), negative_slope=self.leaky_relu_slope))
            
        return self.dropout[1](self.dense_output_nonlinearity(self.ff_out(x)))

    
