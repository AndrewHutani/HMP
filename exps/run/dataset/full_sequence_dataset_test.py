from __future__ import division

import numpy as np
import pickle

from torch.utils.data import Dataset

import utils.config as config
import utils.constants as constants

class BaseDataset_test(Dataset):
    def __init__(self, dataset, dataset_paths, hist_length, filter_str=None):
        super(BaseDataset_test, self).__init__()
        
        dataset_path = dataset_paths[dataset]
        with open(dataset_path, 'rb') as f:
            label = pickle.load(f)
        if filter_str is not None:
            label = [path for path in label if filter_str.lower() in path.lower()]

        # Group by base name (everything before last underscore)
        from collections import defaultdict
        self.groups = defaultdict(list)
        for path in label:
            # Remove .npy if present
            if path.endswith('.npy'):
                path = path[:-4]
            # Split by underscores, join all except last part (the index)
            base_name = '_'.join(path.split('_')[:-1])
            self.groups[base_name].append(path)
        self.group_keys = list(self.groups.keys())
        self.hist_length = hist_length

    def __getitem__(self, index):
        group = self.groups[self.group_keys[index]]
        q_list, shape_list, gender_id_list = [], [], []
        file_paths = []
        for trunk_path in sorted(group, key=lambda x: int(x.split('_')[-1])):  # Sort by index
            anno = np.load(trunk_path+'.npy')
            file_paths.append(trunk_path + '.npy')
            q_list.append(anno[(config.hist_length-self.hist_length):, :63])
            shape_list.append(anno[(config.hist_length-self.hist_length):, 63:63+10])
            gender_id_list.append(anno[(config.hist_length-self.hist_length):, 63+10])

        Y = {}
        Y['q'] = np.concatenate(q_list, axis=0)
        Y['shape'] = np.concatenate(shape_list, axis=0)
        Y['gender_id'] = np.concatenate(gender_id_list, axis=0)
        Y['file_paths'] = file_paths
        return Y

    def __len__(self):
        return len(self.group_keys)