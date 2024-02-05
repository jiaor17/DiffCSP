import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import pdb

import os

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

train_dist = {
    'perov' : [0, 0, 0, 0, 0, 1],
    'carbon' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301]
}

def diffusion(loader, energy, uncond, step_lr, aug):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    batch = next(iter(loader)).to(energy.device)

    all_crystals = []
    for i in range(1,11):
        print(f'Optimize from T={i*100}')
        outputs, _ = energy.sample(batch, uncond, step_lr = step_lr, diff_ratio = i/10, aug = aug)
        all_crystals.append(outputs)

    res = {k: torch.cat([d[k].detach().cpu() for d in all_crystals], dim=0).unsqueeze(0) for k in
        ['frac_coords', 'atom_types', 'num_atoms', 'lattices']}

    lengths, angles = lattices_to_params_shape(res['lattices'])
    

    return res['frac_coords'], res['atom_types'], lengths, angles, res['num_atoms']


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, loader, cfg = load_model(
        model_path, load_data=True)

    uncond_path = Path(args.uncond_path)

    uncond, _, cfg = load_model(
        uncond_path, load_data=False)    

    if torch.cuda.is_available():
        model.to('cuda')
        uncond.to('cuda')

    print('Evaluate the diffusion model.')

    start_time = time.time()
    (frac_coords, atom_types, lengths, angles, num_atoms) = diffusion(loader, model, uncond, args.step_lr, args.aug)

    if args.label == '':
        gen_out_name = 'eval_opt.pt'
    else:
        gen_out_name = f'eval_opt_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
    }, model_path / gen_out_name)
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--uncond_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--aug', default=50, type=float)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
