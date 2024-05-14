import math
import pickle
import copy
import collections
import functools
import os
import json
import time
import datetime
import numpy as np
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import random as py_random
import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn

from jactorch.utils.meta import as_tensor, as_float, as_cpu
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor

from difflogic.cli import format_args
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase
from IPython.core.debugger import Pdb

logger = get_logger(__file__)

class SudokuConvNet(nn.Module):
    def __init__(self, args):
        super(SudokuConvNet, self).__init__()
        self.args = args

        self.layers = [100,64,32,32] 
        num_input_channels = self.args.sudoku_num_steps
        if (self.args.is_encoder_decoder) and (self.args.loss_on_encoder):
            num_input_channels = 2*num_input_channels


        self.add_module("conv_0", nn.Conv2d(
            num_input_channels, self.layers[0], kernel_size=3, padding=1))
        for i in range(1, len(self.layers)):
            self.add_module("conv_{}".format(i), nn.Conv2d(
                self.layers[i-1], self.layers[i], kernel_size=3, padding=1))
        self.add_module("conv_{}".format(len(self.layers)), nn.Conv2d(
            self.layers[-1], 1, kernel_size=3, padding=1))

        self.linear = nn.Linear(81, 1)

        if args.use_gpu:
            self = self.cuda()

    def forward(self, feed_dict, y_hat, additional_info=None):
        feed_dict = GView(feed_dict)
        target = feed_dict["target"].long()
        sudoku_num_steps= y_hat.shape[-1]
        # y_hat has shape exp_batch_size x 10 x 81 x num_steps
        # x has shape exp_batch_size x 81 x num_steps
        #Pdb().set_trace()
        x = y_hat.argmax(dim=1).long()
        x = (x == target.unsqueeze(-1).expand(len(y_hat),
                                        81, sudoku_num_steps)).float()


        #print("error at last layer", x[:,:,-1].sum(dim=-1))
        #print("count: ", feed_dict['count'])
        # shuffle dimensions to make it exp_batch_size x num_steps x 81
        # reshape it to exp_batch_size x num_steps x 9 x 9
        x = x.transpose(1, 2).view(-1, sudoku_num_steps, 9, 9)

        #Pdb().set_trace()
        for i in range(len(self.layers)+1):
            x = torch.nn.functional.gelu(self._modules["conv_{}".format(i)](x))
        x = x.view(-1, 81)
        #print("min x: ", x.min(), "max x: ", x.max())
        return {'latent_z': self.linear(x)}



class DeterministicLatentModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dummy_parameter = nn.Parameter(torch.ones(1))
        logger.info("Returning deterministic latent model")

    def forward(self, feed_dict, y_hat, additional_info=None):
        constant = (self.dummy_parameter + 100)/(self.dummy_parameter + 100)
        return dict(latent_z=constant*self.args.latent_annealing*feed_dict['query'].size(1)*feed_dict['loss'].unsqueeze(1))


class EpsilonGreedyLatentModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dummy_parameter = nn.Parameter(torch.ones(1))
        logger.info("Returning deterministic epsilon greedy latent model")

    def forward(self, feed_dict, y_hat, additional_info=None):
        constant = (self.dummy_parameter + 100)/(self.dummy_parameter + 100)
        # Pdb().set_trace()
        return dict(latent_z=constant*feed_dict['minloss_eg_prob'].unsqueeze(1))


class LatentNLMModel(nn.Module):
    """ The model for latent variable """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feature_axis = 0
        # inputs
        input_dims = [0 for _ in range(args.latent_breadth + 1)]
        if args.task_is_nqueens:
            input_dims[1] = 1
            input_dims[2] = 4
        elif args.task_is_futoshiki:
            input_dims[1] = 3
            input_dims[2] = 3

        self.features = LogicMachine.from_args(
            input_dims, args.latent_attributes, args, prefix='latent')
        output_dim = self.features.output_dims[self.feature_axis]

        target_dim = 1
        # nothing but MLP with sigmoid
        self.pred = LogitsInference(output_dim, target_dim, [])
        self.latent_breadth = args.latent_breadth
        self.task_is_futoshiki = args.task_is_futoshiki
        self.task_is_sudoku = args.task_is_sudoku

    def forward(self, feed_dict, y_hat, additional_info=None):
        feed_dict = GView(feed_dict)
        # Pdb().set_trace()
        relations = feed_dict.relations.float()

        batch_size, nr = relations.size()[:2]

        #states = feed_dict.query.float()
        # @TODO : should we give x as input as well?
        if self.task_is_futoshiki:
            states = torch.stack([y_hat - feed_dict.target.float(), feed_dict.query[:,
                                                                                    :, 1].float(), feed_dict.query[:, :, 2].float()], 2)
        elif self.task_is_sudoku:
            states = y_hat.transpose(
                1, 2) - torch.nn.functional.one_hot(feed_dict.target.long(), 10).float()
        else:
            states = (y_hat - feed_dict.target.float()).unsqueeze(2)
        #

        inp = [None for _ in range(self.latent_breadth + 1)]
        inp[1] = states
        inp[2] = relations

        depth = None
        feature = self.features(inp, depth=None)[self.feature_axis]

        latent_z = self.pred(feature)
        return dict(latent_z=latent_z)



