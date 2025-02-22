#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os.path as osp
import glob
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

import data.data_transform as Transforms
from utils.config import cfg

shufflepoints = Transforms.ShufflePoints(num=cfg.DATASET.POINT_NUM)

class WaymoFlow(Dataset):
    def __init__(self, dataset_root, num_points=1024, partition='train'):
        self.dataset_root = dataset_root
        self.num_points = num_points
        self.partition = partition

        import pickle
        if osp.exists('data_list'+f'_{self.partition}'):
            with open ('data_list'+f'_{self.partition}', 'rb') as fp:
                self.data_list = pickle.load(fp)
        else:
            self.data_list = sorted(glob.glob(osp.join(self.dataset_root, self.partition) + '/previous/**/*.npy'))
            with open('data_list'+f'_{self.partition}', 'wb') as fp:
                pickle.dump(self.data_list, fp)


    def get_data_label(self, data_name):


        if 'TYPE_VEHICLE' in data_name:
            label = 1
        elif 'TYPE_PEDESTRIAN' in data_name:
            label = 2
        elif 'TYPE_SIGN' in data_name:
            label = 3
        return label



    def load_data(self, index):
        ref_frame = self.data_list[index]
        src_frame = self.data_list[index].replace('previous', 'current')
        transform = self.data_list[index].replace('previous', 'gt_poses')
        ref_frame = np.load(ref_frame)
        src_frame = np.load(src_frame)
        transform = np.load(transform)
        label = self.get_data_label(self.data_list[index])
        return ref_frame, src_frame, transform, label

    def __getitem__(self, item):
        points_ref_raw, points_src_raw, transform_gt, label = self.load_data(item)

        # indices_ref = np.random.choice(points_ref_raw.shape[0], self.num_points, replace=True)
        # indices_src = np.random.choice(points_src_raw.shape[0], self.num_points, replace=True)

        # points_ref, points_src = points_ref_raw[indices_ref], points_src_raw[indices_src]

        sample = {
            'points_src_raw':points_src_raw,
            'points_ref_raw':points_ref_raw,
            # 'points_src':points_src, 
            # 'points_ref':points_ref, 
            'label': np.array(label, dtype=np.float32), 
            'idx': np.array(item, dtype=np.float32),
            'transform_gt': np.array(transform_gt, dtype=np.float32)
            }

        sample = shufflepoints(sample)
 
        transform_gt = sample['transform_gt']
        num_src = len(points_src_raw)
        num_ref = len(points_ref_raw)
        # num_src = len(points_src)
        # num_ref = len(points_ref)

        ret_dict = {
            'points': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
            'num': [torch.tensor(x) for x in [num_src, num_ref]],
            'transform_gt': torch.Tensor(transform_gt.astype('float32')),
            'perm_mat_gt': torch.tensor(sample['perm_mat'].astype('float32')),
            'overlap_gt': [torch.Tensor(x) for x in [sample['src_overlap_gt'], sample['ref_overlap_gt']]],
            'label': torch.tensor(sample['label']),
            'points_src_raw': torch.Tensor(sample['points_src_raw']),
            'points_ref_raw': torch.Tensor(sample['points_ref_raw'])
        }

        return ret_dict

    def __len__(self):
        return len(self.data_list)


def get_datasets(partition='train'):
    dataset_root = cfg.DATASET.ROOT
    num_points = cfg.DATASET.POINT_NUM
    datasets = WaymoFlow(dataset_root, num_points, partition)
    return datasets


def collate_fn(data: list):
    """
    Create mini-batch data2d for training.
    :param data: data2d dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    return ret


def get_dataloader(dataset, phase, shuffle=False):
    if phase == 'test':
        batch_size = cfg.DATASET.TEST_BATCH_SIZE
    else:
        batch_size = cfg.DATASET.TRAIN_BATCH_SIZE
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=cfg.DATALOADER_NUM,
                                       collate_fn=collate_fn, pin_memory=False)
