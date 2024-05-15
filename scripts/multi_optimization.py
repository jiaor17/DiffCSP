import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
import torch.nn
import torch.nn.functional as F
from torch.autograd import grad
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, lattice_params_to_matrix_torch, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map
MAX_ATOMIC_NUM=100

import pdb

import os

class SampleDataset(Dataset):

    def __init__(self, ori_dataset, total_num, max_atom = 80):
        super().__init__()
        self.total_num = total_num
        self.max_atom = max_atom
        self.distribution = self.get_distribution(ori_dataset)
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)

    def get_distribution(self, ori_dataset):
        print("Calculating data distribution from training set.")
        nums = [0 for i in range(self.max_atom + 1)]
        for i in tqdm(range(len(ori_dataset))):
            n_i = ori_dataset[i].num_atoms
            if n_i <= self.max_atom:
                nums[n_i] += 1
        return np.array(nums).astype(np.float32) / sum(nums)

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
        )
        return data
    
def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
        
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)

class IndependentMultipleOptimization(object):

    def __init__(self, uncond_model, cond_models):

        self.uncond_model = uncond_model
        self.cond_models = cond_models
        self.beta_scheduler = uncond_model.beta_scheduler
        self.sigma_scheduler = uncond_model.sigma_scheduler
        self.device = uncond_model.device

    def optimization_from_random(self, loader, step_lr, augs):

        all_crystals = []
        for idx, batch in enumerate(loader):
            print(f'Batch {idx} / {len(loader)}')
            if torch.cuda.is_available():
                batch.cuda()
            outputs, _ = self.optimization_batch(batch, step_lr = step_lr, augs = augs)
            all_crystals.append(outputs)
        res = {k: torch.cat([d[k].detach().cpu() for d in all_crystals], dim=0).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lattices']}
        lengths, angles = lattices_to_params_shape(res['lattices'])
        return res['frac_coords'], res['atom_types'], lengths, angles, res['num_atoms']

    def optimization_from_templates(self, loader, step_lr, augs):

        all_crystals = [[] for i in range(10)]
        for idx, batch in enumerate(loader):
            if torch.cuda.is_available():
                batch.cuda()
            for i in range(1,11):
                print(f'Batch {idx} / {len(loader)}, Optimize from T={i*100}')
                outputs, _ = self.optimization_batch(batch, step_lr = step_lr, diff_ratio = i/10, augs = augs)
                all_crystals[i - 1].append(outputs)

        ans = []
        for i in range(10):
            res = {k: torch.cat([d[k].detach().cpu() for d in all_crystals[i]], dim=0).unsqueeze(0) for k in
                ['frac_coords', 'atom_types', 'num_atoms', 'lattices']}
            lengths, angles = lattices_to_params_shape(res['lattices'])
            ans.append({
                'frac_coords': res['frac_coords'].unsqueeze(0),
                'atom_types': res['atom_types'].unsqueeze(0),
                'lengths': lengths.unsqueeze(0),
                'angles': angles.unsqueeze(0), 
                'num_atoms': res['num_atoms'].unsqueeze(0)
            })

        final_ans = {k: torch.cat([d[k] for d in ans], dim=0) for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}

        
        return final_ans['frac_coords'], final_ans['atom_types'], final_ans['lengths'], final_ans['angles'], final_ans['num_atoms']

    def optimization_batch(self, batch, step_lr, augs, diff_ratio=1.0):

        assert len(augs) == len(self.cond_models)

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords

            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]

            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)

            sigmas = self.sigma_scheduler.sigmas[time_start]

            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t

        else:
            time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.uncond_model.time_embedding(times)

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)


            pred_l, pred_x, pred_t = self.uncond_model.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
            l_t_minus_05 = l_t
            t_t_minus_05 = t_t


            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x, pred_t = self.uncond_model.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            grads_t, grads_x, grads_l = torch.zeros_like(t_T), torch.zeros_like(x_T), torch.zeros_like(x_T),
            with torch.enable_grad():
                for (cond_model, aug) in zip (self.cond_models, augs):
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = cond_model.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)
                        grads_t = grads_t + grad_t * aug
                        grads_l = grads_l + grad_l * aug
                        grads_x = grads_x + grad_x * aug

            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * grads_x + std_x * rand_x 
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * grads_l + sigmas * rand_l 
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * grads_t + sigmas * rand_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1

        return res, traj_stack



def main(args):
    # load_data if do reconstruction.

    uncond_path = Path(args.uncond_path)

    uncond, _, cfg = load_model(
        uncond_path, load_data=False)   

    conds = []
    used_loader = None
    for cond_path in args.cond_paths:
        model_path = Path(cond_path)
        model, loader, cfg = load_model(
            model_path, load_data=used_loader is None, testing=False) 
        if used_loader is None:
            used_loader = loader
        conds.append(model)

    if torch.cuda.is_available():
        uncond.to('cuda')
        for model in conds:
            model.to('cuda')

    train_set = used_loader[0].dataset

    sample_set = SampleDataset(train_set, args.total_num, args.max_atom)

    sample_loader = DataLoader(sample_set, batch_size = args.batch_size)

    print('Evaluate the diffusion model.')

    optimizer = IndependentMultipleOptimization(uncond, conds)

    if len(args.augs) == 1:
        augs = [args.augs[0] for i in range(len(conds))]

    else:
        augs = args.augs

    (frac_coords, atom_types, lengths, angles, num_atoms) = optimizer.optimization_from_random(sample_loader, args.step_lr, augs)

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
    parser.add_argument('--cond_paths', nargs='+', required=True)
    parser.add_argument('--uncond_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--augs', nargs='+', default=[50], type=float)
    parser.add_argument('--total_num', default=500, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--max_atom', default=80, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
