import pandas as pd
import numpy as np
import os
import argparse

def split_df(ori_df, random_seed=42, val_ratio=0.1, test_ratio=0.1):
    num = len(ori_df)
    ids = list(range(num))
    np.random.seed(random_seed)
    np.random.shuffle(ids)
    val_num = int(num * val_ratio)
    test_num = int(num * test_ratio)
    train_num = num - val_num - test_num
    train_ids = ids[:train_num]
    val_ids = ids[train_num:train_num + val_num]
    test_ids = ids[-test_num:]
    return ori_df.iloc[train_ids], ori_df.iloc[val_ids], ori_df.iloc[test_ids]

def get_cifs(cif_dir, cif_list):
    res = []
    for cif in cif_list:
        with open(os.path.join(cif_dir, cif),'r') as f:
            l = f.read()
            res.append(l)
    return res

def main(args):
    ori_csv = os.path.join(args.dir, args.csv)
    ori_df = pd.read_csv(ori_csv, header=0)
    train_df, val_df, test_df = split_df(ori_df, random_seed=args.random_seed, val_ratio=args.val, test_ratio=args.test)
    train_df['cif'] = get_cifs(os.path.join(args.dir,'cif'), train_df['name'])
    val_df['cif'] = get_cifs(os.path.join(args.dir,'cif'), val_df['name'])
    test_df['cif'] = get_cifs(os.path.join(args.dir,'cif'), test_df['name'])
    train_df['material_id'] = [_.split('.')[0] for _ in train_df['name']]
    val_df['material_id'] = [_.split('.')[0] for _ in val_df['name']]
    test_df['material_id'] = [_.split('.')[0] for _ in test_df['name']]
    train_df.to_csv(os.path.join(args.dir, 'train.csv'))
    val_df.to_csv(os.path.join(args.dir, 'val.csv'))
    test_df.to_csv(os.path.join(args.dir, 'test.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--val', default=0.1, type=float)
    parser.add_argument('--test', default=0.1, type=float)
    args = parser.parse_args()
    main(args)