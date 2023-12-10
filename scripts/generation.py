import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list, recommand_step_lr

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

train_dist = {
    'perov_5' : [0, 0, 0, 0, 0, 1],
    'carbon_24' : [0.0,
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
    'mp_20' : [0.0,
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


def diffusion(loader, model, step_lr):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

class SampleDataset(Dataset):

    def __init__(self, dataset, total_num):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon_24'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['gen'][args.dataset]

    print(step_lr)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, step_lr)

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

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
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
