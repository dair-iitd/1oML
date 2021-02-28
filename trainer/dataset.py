#! /usr/bin/env python3
import numpy as np
import itertools
import math
import copy
import pickle

import torch
from torch.utils.data.dataset import Dataset
from torch.distributions.categorical import Categorical

import jacinle.random as random
from jacinle.logging import get_logger, set_output_file

TRAIN = 0
DEV = 1
TEST = 2

logger = get_logger(__file__)

__all__ = [
    'NQueensDataset', 'FutoshikiDataset', 'SudokuDataset'
]

class NQueensDataset(Dataset):
    """The dataset for nqueens tasks."""

    def __init__(self,
                 epoch_size,
                 n=10,
                 random_seed=42,
                 arbit_solution=False,
                 train_dev_test=TRAIN,
                 data_file=None,
                 data_sampling='rs'):
        super().__init__()

        self._epoch_size = epoch_size
        self._n = n
        self.arbit_solution = arbit_solution
        self.mode = train_dev_test
        self.data_sampling = data_sampling

        self.relations = self.get_relations()
        print("In constructor. Size: {}".format(n))
        outfile = data_file
        #
        with open(outfile, "rb") as f:
            self.dataset = pickle.load(f)

        self.max_count = 0
        self.unique_indices = []
        self.ambiguous_indices = []
        for i, data in enumerate(self.dataset):
            self.max_count = max(self.max_count, data["count"])
            if data["count"] == 1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)

        np.random.seed(random_seed)
        self.reset_sampler(data_sampling)
    
    def get_relations(self):
        get_xy = lambda a,n: (a//n, a%n)
        n = self._n
        board_size = n*n
        rows = np.zeros((board_size, board_size))
        cols = np.zeros((board_size, board_size))
        diagonals = np.zeros((board_size, board_size))
        off_diagonals = np.zeros((board_size, board_size))
        for i in range(board_size):
            for j in range(board_size):
                row1, col1 = get_xy(i, n)
                row2, col2 = get_xy(j, n)
                if row1 == row2:
                    rows[i, j] = 1
                if col1 == col2:
                    cols[i, j] = 1
                if (row2 - row1) == (col2 - col1):
                    diagonals[i, j] = 1
                if (row2 - row1) == (col1 - col2):
                    off_diagonals[i, j] = 1

        return np.stack([rows, cols, diagonals, off_diagonals]).swapaxes(0, 2)

    def reset_sampler(self, data_sampling):
        self.data_sampling = data_sampling
        if data_sampling == 'rsxy':
            logger.info("Sampling uniformly from (x,y) tuples")
            self.sampler = Categorical(probs=torch.tensor(
                [x['count'] for x in self.dataset]).float())
        else:
            self.sampler = Categorical(probs=torch.tensor(
                [1.0 for _ in self.dataset]).float())

    def pad_set(self, target_set):
        pad_counter = self.max_count - len(target_set)
        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand() < imbalance_ratio:
            ind = np.random.choice(self.ambiguous_indices)
        else:
            ind = np.random.choice(self.unique_indices)
        return ind

    def __getitem__(self, item):
        #ind = np.random.randint(0,len(self.dataset))
        ind = self.sampler.sample().item()
        if self.mode == TRAIN:
            if self.data_sampling == "unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling == "ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling == "one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling == "two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling == "three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling == "four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item % len(self.dataset)

        data = self.dataset[ind]

        if len(data["query"].shape) == 1:
            data["query"] = np.expand_dims(data["query"], 1)
        if self.mode == TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(
                len(data["target_set"]))]
        #
        data["target_set"] = self.pad_set(data["target_set"])
        data['mask'] = np.array([1 for _ in range(
            data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])])
        # Pdb().set_trace()
        data["relations"] = self.relations
        data['ind'] = ind
        if isinstance(data["qid"], tuple):
            data["qid"] = np.array([data["qid"][0]]+list(data["qid"][1]))
        return data

    def __len__(self):
        if self.mode == TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)


class FutoshikiDataset(Dataset):
    """The dataset for futoshiki tasks."""

    def __init__(self,
                 epoch_size,
                 n=10,
                 data_size=-1,
                 random_seed=42,
                 arbit_solution=False,
                 train_dev_test=TRAIN,
                 data_file=None,
                 data_sampling='rs', args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self._n = n
        self.arbit_solution = arbit_solution
        self.mode = train_dev_test
        self.data_sampling = data_sampling

        self.relations = self.get_relations()
        print("In constructor. Size: {}".format(n))
        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test == DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        outfile = data_file
        #
        logger.info("data file : {}".format(outfile))
        # Pdb().set_trace()
        with open(outfile, "rb") as f:
            self.dataset = pickle.load(f)

        if data_size != -1:
            self.dataset = self.dataset[:data_size]
        #
        self.max_count = 0
        self.unique_indices = []
        self.ambiguous_indices = []
        for i, data in enumerate(self.dataset):
            if 'count' in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                data['count'] = this_count
            self.max_count = max(self.max_count, this_count)
            if this_count == 1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)
        np.random.seed(random_seed)
        self.reset_sampler(data_sampling)

    def reset_sampler(self, data_sampling):
        self.data_sampling = data_sampling
        if data_sampling == 'rsxy':
            logger.info("Sampling uniformly from (x,y) tuples")
            self.sampler = Categorical(probs=torch.tensor(
                [x['count'] for x in self.dataset]).float())
        else:
            self.sampler = Categorical(probs=torch.tensor(
                [1.0 for _ in self.dataset]).float())

    def get_relations(self):
        n = self._n
        n2 = self._n**2
        n3 = self._n**3
        relations = np.zeros((n3, n3, 3))

        for x in range(n3):
            row = int(x/n2)
            col = int((x % n2)/n)
            num = int(x % n2) % n

            for y in range(n):
                # cell constraints
                relations[x][row*n2+col*n+y][0] = 1

                # row constraints
                relations[x][y*n2+col*n+num][1] = 1

                # column constraints
                relations[x][row*n2+y*n+num][2] = 1
        return relations

    def pad_set(self, target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand() < imbalance_ratio:
            ind = np.random.choice(self.ambiguous_indices)
        else:
            ind = np.random.choice(self.unique_indices)
        return ind

    def __getitem__(self, item):
        # Pdb().set_trace()
        #ind = np.random.randint(0,len(self.dataset))
        ind = self.sampler.sample().item()
        # print(ind)
        if self.mode == TRAIN:
            if self.data_sampling == "unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling == "ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling == "one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling == "two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling == "three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling == "four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item % len(self.dataset)

        data = self.dataset[ind]

        if self.mode == TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(
                data['count'])]

        data["target_set"] = self.pad_set(data["target_set"])

        data['n'] = self._n
        data['is_ambiguous'] = int(data['count'] > 1)
        data['qid'] = np.array([ind])
        data['ind'] = ind
        data['mask'] = np.array([1 for _ in range(
            data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])])

        if self.args.model != 'satnet' or self.args.latent_model == 'nlm':
            data["relations"] = self.relations
        if self.args.model == 'satnet':
            data['gtlt'] = np.concatenate(
                (data['query'][::self._n, 1], data['query'][::self._n, 2]), axis=0)
        return data

    def __len__(self):
        if self.mode == TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)


class SudokuDataset(Dataset):
    """The dataset for sudoku tasks."""

    def __init__(self,
                 epoch_size,
                 data_size=-1,
                 arbit_solution=False,
                 train_dev_test=TRAIN,
                 data_file=None,
                 data_sampling='rs', args=None):
        super().__init__()
        self.args = args
        self._epoch_size = epoch_size
        self.arbit_solution = arbit_solution
        self.mode = train_dev_test
        self.data_sampling = data_sampling
        self._n = 81
        print("In constructor.  {}".format(args.task))
        if train_dev_test == TRAIN:
            mode = 'train'
        elif train_dev_test == DEV:
            mode = 'val'
        elif train_dev_test == TEST:
            mode = 'test'

        outfile = data_file
        #
        logger.info("data file : {}".format(outfile))
        # Pdb().set_trace()
        with open(outfile, "rb") as f:
            self.dataset = pickle.load(f)

        if data_size != -1:
            self.dataset = self.dataset[:data_size]
        #
        np.random.seed(args.seed)
        self.max_count = args.solution_count
        self.unique_indices = []
        self.ambiguous_indices = []
        for i, data in enumerate(self.dataset):
            data['query'] = (data['query']).astype(int)
            if len(data["target_set"]) > self.max_count:
                self.dataset[i]["target_set"] = data["target_set"][:self.max_count]
                self.dataset[i]["count"] = self.max_count
            if 'count' in data:
                this_count = data['count']
            else:
                this_count = data['target_set'].shape[0]
                self.dataset[i]['count'] = this_count
            if this_count == 1:
                self.unique_indices.append(i)
            else:
                self.ambiguous_indices.append(i)
        self.max_count += 1
        self.reset_sampler(data_sampling)

    def reset_sampler(self, data_sampling):
        self.data_sampling = data_sampling
        if data_sampling == 'rsxy':
            logger.info("Sampling uniformly from (x,y) tuples")
            self.sampler = Categorical(probs=torch.tensor(
                [x['count'] for x in self.dataset]).float())
        else:
            self.sampler = Categorical(probs=torch.tensor(
                [1.0 for _ in self.dataset]).float())

    def pad_set(self, target_set):
        pad_counter = self.max_count - len(target_set)
        if pad_counter < 0:
            return target_set[:self.max_count]

        return_set = list(target_set)
        return_set.extend([target_set[-1] for _ in range(pad_counter)])
        return np.array(return_set)

    def sample_imbalance(self, imbalance_ratio):
        if np.random.rand() < imbalance_ratio:
            ind = np.random.choice(self.ambiguous_indices)
        else:
            ind = np.random.choice(self.unique_indices)
        return ind

    def __getitem__(self, item):
        # Pdb().set_trace()
        #ind = np.random.randint(0,len(self.dataset))
        ind = self.sampler.sample().item()
        # print(ind)
        if self.mode == TRAIN:
            if self.data_sampling == "unique":
                ind = self.sample_imbalance(0)
            elif self.data_sampling == "ambiguous":
                ind = self.sample_imbalance(1)
            elif self.data_sampling == "one-one":
                ind = self.sample_imbalance(0.5)
            elif self.data_sampling == "two-one":
                ind = self.sample_imbalance(0.33)
            elif self.data_sampling == "three-one":
                ind = self.sample_imbalance(0.25)
            elif self.data_sampling == "four-one":
                ind = self.sample_imbalance(0.20)
        else:
            ind = item % len(self.dataset)

        data = self.dataset[ind]

        if self.mode == TRAIN and self.arbit_solution:
            data["target"] = data["target_set"][0]
        else:
            data["target"] = data["target_set"][np.random.randint(
                data['count'])]

        data["target_set"] = self.pad_set(data["target_set"])

        data['n'] = self._n
        data['is_ambiguous'] = int(data['count'] > 1)
        data['qid'] = np.array([ind])
        data['ind'] = ind
        data['mask'] = np.array([1 for _ in range(
            data['count'])] + [0 for _ in range(data['target_set'].shape[0] - data['count'])])

        return data

    def __len__(self):
        if self.mode == TRAIN:
            return self._epoch_size
        else:
            return len(self.dataset)
