import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import h5py
import psutil
import sys
import time



emotion_dic = {
    'neu': 0,
    'neutral': 0,
    'hap': 1,
    'happy': 1,
    'joy': 1,
    'sad': 2,
    'sadness': 2,
    'ang': 3,
    'anger': 3}

class MyIterableDataset(IterableDataset):
    def __init__(self, file_path, dataset_mode, batch_size=32, shuffle=False):
        self.file_path = file_path
        self.dataset_mode = dataset_mode
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # If not using multiple workers, just process the entire dataset
            start_idx = 0
            end_idx = None
        else:
            # If using multiple workers, partition the dataset
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            with h5py.File(self.file_path, 'r') as f:
                a = f[self.dataset_mode]['A_feat']
                n_samples = len(a)
                chunk_size = (n_samples + num_workers - 1) // num_workers
                start_idx = worker_id * chunk_size
                end_idx = min((worker_id + 1) * chunk_size, n_samples)

        with h5py.File(self.file_path, 'r') as f:
            a = f[self.dataset_mode]['A_feat']
            v = f[self.dataset_mode]['V_feat']
            l = f[self.dataset_mode]['L_feat']
            emotion = f[self.dataset_mode]['emotion']
            # emotion = emotion.decode('utf-8')
            # e = torch.tensor([emotion_dic[emotion]])


            n_samples = len(a)
            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)
            indices = indices[start_idx:end_idx] if end_idx is not None else indices
            for idx in range(0, len(indices), self.batch_size):
                batch_indices = sorted(indices[idx:idx+self.batch_size])  #indices[idx:idx+self.batch_size]
                batch_a = torch.from_numpy(a[batch_indices]).float()
                batch_v = torch.from_numpy(v[batch_indices]).float()
                batch_l = torch.from_numpy(l[batch_indices]).float()

                batch_et = emotion[batch_indices]
                batch_et = [i.decode('utf-8') for i in batch_et]
                batch_e = torch.tensor([emotion_dic[i] for i in batch_et])

                batch = {'A_feat': batch_a, 'V_feat': batch_v, 'L_feat': batch_l, 'label': batch_e}
                yield batch

#############################################################################################

################################################################################################


