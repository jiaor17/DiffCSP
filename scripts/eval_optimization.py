from collections import Counter
import argparse
import os
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from torch_geometric.data import Data, Batch, DataLoader

from pyxtal import pyxtal

import torch

import pickle

import sys
sys.path.append('.')


from eval_utils import load_model, load_data, smact_validity, structure_validity, load_config, get_crystals_list

from diffcsp.pl_data.dataset import TensorCrystDataset
from diffcsp.pl_data.datamodule import worker_init_fn


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')


chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

def prop_model_eval(model_path, crystal_array_list):


    model, _, _ = load_model(model_path)
    cfg = load_config(model_path)

    dataset = TensorCrystDataset(
        crystal_array_list, cfg.data.niggli, cfg.data.primitive,
        cfg.data.graph_method, cfg.data.preprocess_workers,
        cfg.data.lattice_scale_method)

    dataset.scaler = model.scaler.copy()

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=256,
        num_workers=0,
        worker_init_fn=worker_init_fn)

    model.eval()

    all_preds = []

    for batch in loader:
        preds = model(batch)

        if model.task == 'regression':
            model.scaler.match_device(preds)
            scaled_preds = model.scaler.inverse_transform(preds)
            all_preds.append(scaled_preds.detach().cpu().numpy())

        elif model.task == 'classification':
            tar_idx = cfg.data.opt_target
            sm_preds = torch.softmax(preds, dim=-1)[tar_idx]
            all_preds.append(sm_preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
    return all_preds.tolist()



class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        self.multiply = int(np.gcd.reduce(counts))
        counts = counts / self.multiply
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())
        

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)

    @property
    def formula(self):
        fml = ''
        for elem, comp in zip(self.elems, self.comps):
            fml = fml + chemical_symbols[elem] + str(comp * self.multiply)
        return fml

def eval_model(model, loader):

    model.eval()

    preds = []
    gts = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        pred = model(batch)
        if model.task == 'classification':
            pred = pred.argmax(dim=-1)
            gt = batch.y.reshape(-1)
        elif model.task == 'regression':
            model.scaler.match_device(pred)
            pred = model.scaler.inverse_transform(pred.reshape(-1))
            gt = model.scaler.inverse_transform(batch.y.reshape(-1))

        preds.append(pred.detach().cpu().numpy())
        gts.append(gt.detach().cpu().numpy())

    preds = np.concatenate(preds,axis=-1)
    gts = np.concatenate(gts,axis=-1)

    metrics = {}

    if model.task == 'regression':
        metrics['pcc'] = np.corrcoef(preds, gts)[0,1]

    elif model.task == 'classification':
        metrics['acc'] = (preds == gts).mean()

    return metrics
    

def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][0],
        data['atom_types'][0],
        data['lengths'][0],
        data['angles'][0],
        data['num_atoms'][0])        

    return crys_array_list


def main(args):
    prop_dir = args.dir
    pred_dir = os.path.join(prop_dir, 'prediction')
    guid_dir = os.path.join(prop_dir, 'guidance')
    res_dir = os.path.join(prop_dir, 'results')
    os.makedirs(res_dir, exist_ok=True)

    with open(os.path.join(res_dir, 'summary.log'), 'w') as f:
        f.write('')

    log_file = open(os.path.join(res_dir, 'summary.log'), 'a')

    # Evaluating the property prediction model

    log_file.write("*" * 15 + " Property Prediction " + "*" * 15 + '\n\n')

    model, loader, cfg = load_model(
        Path(pred_dir), load_data=True)
    

    if torch.cuda.is_available():
        model.to('cuda')

    pred_res = eval_model(model, loader)

    for m in pred_res:
        log_file.write(f"Test {m}: {pred_res[m]:.4f}\n\n")

    # Evaluating the optimized samples

    log_file.write("*" * 15 + " Optimization " + "*" * 15 + '\n\n')

    crys_list = get_crystal_array_list(os.path.join(guid_dir,f'eval_opt.pt'))

    crys = p_map(lambda x:Crystal(x), crys_list)

    valid_crys = [c for c in crys if c.valid]

    props = np.ones(len(crys)) * np.nan

    pred_props = prop_model_eval(Path(pred_dir), [c.dict for c in valid_crys])

    mask = np.array([i for (i,c) in enumerate(crys) if c.valid])

    props[mask] = pred_props

    prop_name = cfg.data.prop

    if model.task == 'classification':
        prop_name = prop_name + f'_probability_{cfg.data.opt_target}'

    df = pd.DataFrame({
        'idx' : list(range(1, len(crys) + 1)),
        prop_name : props,
        'valid' : [c.valid for c in crys],
        'formula' : [c.formula for c in crys]
    })

    ascending = (cfg.data.task == 'regression') and (cfg.data.opt_target == -1)

    df.sort_values(by=[prop_name], ascending=ascending, inplace=True, ignore_index=True)

    df.to_csv(os.path.join(res_dir, 'results.csv'))

    cif_dir = os.path.join(res_dir, 'cif')
    os.makedirs(cif_dir, exist_ok=True)


    for i,cry in enumerate(crys):
        tar_file = os.path.join(cif_dir, f"{i+1}_{cry.formula}.cif")
        try:
            writer = CifWriter(cry.structure)
            writer.write_file(tar_file)
        except:
            with open(tar_file, 'w') as f:
                f.write(f"{i+1} Error Structure.")

    log_file.write("Top-5 Results: \n")

    for i in range(5):
        log_file.write(f"{df['idx'][i]}-{df['formula'][i]}: {df[prop_name][i]:.4f}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    main(args)