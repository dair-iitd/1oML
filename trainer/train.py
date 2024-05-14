#! /usr/bin/env python3
import sys
import logging
import math
import pickle
import copy
import collections
import functools
import os
import json
import time
import datetime
import warnings
import numpy as np
import random as py_random
from collections import Counter
from tqdm import tqdm
from IPython.core.debugger import Pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import jacinle.random as random
import jacinle.io as io
import jactorch.nn as jacnn
from jactorch.utils.meta import as_tensor, as_float, as_cpu
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.data.dataloader import JacDataLoader
from jactorch.optim.quickaccess import get_optimizer
from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor

from difflogic.cli import format_args
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase

from dataset import NQueensDataset, FutoshikiDataset, SudokuDataset
import models
import utils
import scheduler

warnings.simplefilter('once')
torch.set_printoptions(linewidth=150)

TASKS = ['nqueens', 'futoshiki', 'sudoku']

parser = JacArgumentParser()

parser.add_argument('--upper-limit-on-grad-norm', type=float, default=1000,
                            metavar='M', help='skip optim step if grad beyond this number')

parser.add_argument('--solution-count', type=int, default=5,
                            metavar='M', help='number at which to cap target-set')

parser.add_argument(
    '--model',
    default='nlm',
    choices=['nlm', 'rrn', 
        'gpt_encoder_sudoku',
        'gpt_encoder_decoder_sudoku'],
    help='model choices, nlm: Neural Logic Machine')

# NLM parameters, works when model is 'nlm'
nlm_group = parser.add_argument_group('Neural Logic Machines')
LogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 4,
        'breadth': 2,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')

nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)


nlm_latent_group = parser.add_argument_group('NLM Latent Model Args')

LogicMachine.make_nlm_parser(
    nlm_latent_group, {
        'depth': 4,
        'breadth': 2,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='latent')

nlm_latent_group.add_argument(
    '--latent-attributes',
    type=int,
    default=10,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)



nlm_latent_group.add_argument('--warmup-epochs', type=int,
                              default=200, metavar='N', help='#of iterations with z = 0')

nlm_latent_group.add_argument('--latent-model', type=str, choices=['conv','nlm','det','eg'],
                              default='nlm', metavar='S', help='which latent model to use when model is rrn')

nlm_latent_group.add_argument('--latent-annealing', type=float,
                                 default=1.0, metavar='N', help='Used when latent model is deterministic, i.e. IExplR')

nlm_latent_group.add_argument('--minloss-eg-eps', type=float,
                                 default=0.05, metavar='N', help='epsilon for minloss eg')

nlm_latent_group.add_argument('--latent-wt-decay', type=float,
                                 default=0, metavar='N', help='wt decay for latent optimizer')


rrn_group = parser.add_argument_group('rrn model specific args')
rrn_group.add_argument('--sudoku-num-steps', type=int,
                       default=32, metavar='N', help='num steps')
rrn_group.add_argument('--sudoku-embed-size', type=int,
                       default=16, metavar='N', help='embed size')
rrn_group.add_argument('--sudoku-hidden-dim', type=int,
                       default=96, metavar='N', help='sudoku hidden dim')

rrn_group.add_argument('--sudoku-do', type=float, default=0.1,
                       metavar='N', help='dropout for msg passing')

rrn_group.add_argument('--sudoku-attention-mask', type=int, default=0,
                       metavar='N', help='should assume sudoku attention mask when training gpt?')
rrn_group.add_argument('--share-decoder-weights', type=int,
                       default=1, metavar='N', help='in t5 model, should decoder wts be shared?')
rrn_group.add_argument('--share-encoder-weights', type=int,
                       default=0, metavar='N', help='in t5 model, should encoder wts be shared?')

rrn_group.add_argument('--loss-on-encoder', type=int,
                       default=0, metavar='N', help='should compute loss using encoder logits?')

rrn_group.add_argument('--is-encoder-decoder', type=int,
                       default=0, metavar='N', help='is it an encoder decoder model?')

# task related

task_group = parser.add_argument_group('Task')
task_group.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')

task_group.add_argument(
    '--train-number',
    type=int,
    default=10,
    metavar='N',
    help='board size of training instances')

data_gen_group = parser.add_argument_group('Data Generation')


data_gen_group.add_argument('--train-data-size', type=int, default=-1,
                            metavar='M', help='size of training data in FutoshikiDataset')

data_gen_group.add_argument('--pretrain-phi', type=int,
                            default=0, help='whether to pretrain phi network  or not')

data_gen_group.add_argument('--min-loss', type=int, default=0,
                            help='compute minimum of loss over possible solutions')

data_gen_group.add_argument('--naive-pll-loss', type=int, default=0,
                            help='compute avg of loss over possible solutions')

data_gen_group.add_argument('--cc-loss', type=int, default=0,
                            help='loss = -log(sum_prob_over_different_targets)')

data_gen_group.add_argument('--arbit-solution', type=int, default=0,
                            help='pick an arbitrary solution from the list of possible solutions')

data_gen_group.add_argument(
    '--train-file', type=str, help="train data file", default='data/nqueens_data_10_5.pkl')

data_gen_group.add_argument('--test-file', type=str, help="test data file")

data_gen_group.add_argument('--hot-data-sampling', type=str,
                            default="rs", help="data sampling strategy when hot",
                            choices=['unique','ambiguous','one-one',
                             'two-one','three-one','four-one','rs'])

data_gen_group.add_argument('--warmup-data-sampling', type=str,
                            default="rs", help="data sampling strategy when in warmup phase",
                            choices=['unique','ambiguous','one-one',
                             'two-one','three-one','four-one','rs','rsxy'])

train_group = parser.add_argument_group('Train')

train_group.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='seed of jacinle.random')

train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')

train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')

train_group.add_argument(
    '--get-optim-from-model',
    type=int,
    default=0,
    help='should ask model class to create an optimizer? required for gpt')


train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')

train_group.add_argument(
    '--lr-hot',
    type=float,
    default=0.001,
    metavar='F',
    help='initial learning rate for hot mode')

train_group.add_argument(
    '--lr-latent',
    type=float,
    default=0.0,
    metavar='F',
    help='initial learning rate for hot mode')


train_group.add_argument(
    '--wt-decay',
    type=float,
    default=0.0,
    metavar='F',
    help='weight decay of learning rate per lesson')

train_group.add_argument(
    '--grad-clip',
    type=float,
    default=1000.0,
    metavar='F',
    help='value at which gradients need to be clipped')

train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for training')

train_group.add_argument(
    '--max-batch-size',
    type=int,
    default=8,
    metavar='N',
    help='batch size for training')

train_group.add_argument(
    '--test-batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for testing')

train_group.add_argument(
    '--reduce-lr',
    type=int,
    default=1,
    metavar='N',
    help='should reduce lr and stop early?')


train_group.add_argument('--latent-dis-prob', type=str, default='softmax',
                         choices={'softmax', 'inverse'}, help=" how to convert distance to prob")

train_group.add_argument('--rl-exploration', type=int, default=0,
                         help="whether to do exploration based on rl agent's policy or choose greedy action")

train_group.add_argument('--exploration-eps', type=float, default=0,
                         help="epsilon for epsilon greedy policy")

train_group.add_argument('--rl-reward', type=str, default='count', choices={
                         'acc', 'count'}, help="rl reward. discrete accuracy or pointwise correct count")

train_group.add_argument('--skip-warmup', type=int, default=0,
                         help="whether to skip warmup if checkkpoint is also given.")

train_group.add_argument('--copy-back-frequency', type=int,
                         default=0, help="frequency at which static model to be updated")

train_group.add_argument('--no-static', type=int,
                         default=0, help="no static model.")

# Note that nr_examples_per_epoch = epoch_size * batch_size
TrainerBase.make_trainer_parser(
    parser, {
        'epochs': 20,
        'epoch_size': 250,
        'test_epoch_size': 1000,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 10,
    })

io_group = parser.add_argument_group('Input/Output')

io_group.add_argument(
    '--dump-dir', type=str, default=None, metavar='DIR', help='dump dir')

io_group.add_argument(
    '--load-checkpoint',
    type=str,
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')


schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=200,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')

schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=1,
    metavar='N',
    help='the interval(number of epochs) to do test')

schedule_group.add_argument(
    '--test-begin-epoch',
    type=int,
    default=0,
    metavar='N',
    help='the interval(number of epochs) after which test starts')

schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')

logger = get_logger(__file__)

glogger = logging.getLogger("grad")
glogger.setLevel(logging.INFO)

args = parser.parse_args()


if args.lr_latent == 0.0:
    args.lr_latent = args.lr_hot

args.use_gpu = args.use_gpu and torch.cuda.is_available()

if args.dump_dir is not None:
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)

    grad_handle = logging.FileHandler(os.path.join(args.dump_dir, 'grad.csv'))
    glogger.addHandler(grad_handle)
    glogger.propagate = False
    glogger.info(
        'epoch,iter,loss,latent_loss,grad_norm_before_clip,grad_norm_after_clip,param_norm_before_clip,lgrad_norm_before_clip,lgrad_norm_after_clip,lparam_norm_before_clip')
else:
    args.checkpoints_dir = None
    args.summary_file = None

if args.seed is not None:
    import jacinle.random as random
    random.reset_global_seed(args.seed)

args.task_is_1d_output = args.task in [
     'nqueens', 'futoshiki']

args.task_is_nqueens = args.task in ['nqueens']
args.task_is_futoshiki = args.task in ['futoshiki']
args.task_is_sudoku = args.task in ['sudoku']


def make_dataset(n, epoch_size, is_train):
    if args.task_is_nqueens:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        #
        return NQueensDataset(epoch_size=epoch_size, n=n,
                              random_seed=args.seed,
                              arbit_solution=args.arbit_solution,
                              train_dev_test=0 if is_train else 2,
                              data_file=data_file,
                              data_sampling=args.warmup_data_sampling)
    elif args.task_is_futoshiki:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        return FutoshikiDataset(epoch_size=epoch_size, n=n,
                                data_size=args.train_data_size if is_train else -1,
                                arbit_solution = args.arbit_solution,
                                random_seed=args.seed,
                                train_dev_test=0 if is_train else 2,
                                data_file=data_file,
                                data_sampling=args.warmup_data_sampling,
                                args=args)
    elif args.task_is_sudoku:
        data_file = args.train_file
        if not is_train:
            data_file = args.test_file
        return SudokuDataset(epoch_size=epoch_size,
                             data_size=args.train_data_size if is_train else -1,
                             arbit_solution=args.arbit_solution,
                             train_dev_test=0 if is_train else 2,
                             data_file=data_file,
                             data_sampling=args.warmup_data_sampling, args=args
                             )
    
def default_reduce_func(k, v):
    return v.mean()

def hook(grad):
    print("grad z latent", grad)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def collapse_tensor(t, count):
    # inverse of expand_tensor
    # same elements are repeated in t. we just have to pick the first one
    orig_batch_size = count.size(0)
    cum_sum = count.cumsum(0)
    return t[cum_sum - 1]


def expand_tensor(t, count):
    # just repeat t[i] count[i] times.
    orig_batch_size = t.size(0)
    rv = []
    final_batch_size = count.sum()
    for i in range(orig_batch_size):
        this_count = count[i]
        rv.append(t[i].unsqueeze(0).expand(this_count, *t[i].size()))
    #
    rv = torch.cat(rv)
    assert rv.size(0) == final_batch_size
    return rv

def unravel_min_indicator(loss_matrix, count):
    # assuming pad_dim as 1
    pad_dim = 1
    orig_batch_size = loss_matrix.size(0)
    final_batch_size = count.sum()
    rv = []
    for i in range(orig_batch_size):
        this_count = count[i]
        this_indicator  = torch.zeros(this_count)
        this_indicator[loss_matrix[i,:this_count].argmin()] = 1
        rv.append(this_indicator)
    #
    rv = torch.cat(rv)
    assert rv.size(0) == final_batch_size
    return rv

def unravel_minloss_epsilon_greedy(loss_matrix, count, epsilon):
    # assuming pad_dim as 1
    pad_dim = 1
    orig_batch_size = loss_matrix.size(0)
    final_batch_size = count.sum()
    rv = []
    #Pdb().set_trace()
    for i in range(orig_batch_size):
        this_count = count[i]
        this_prob = torch.zeros(this_count,device=loss_matrix.device)
        this_prob[loss_matrix[i,:this_count].argmin()] = 1
        this_prob = this_prob*(1 - epsilon) + epsilon/this_count.float()
        rv.append(this_prob)
    #
    rv = torch.cat(rv)
    assert rv.size(0) == final_batch_size
    return rv




def unravel_tensor(target_set, count):
    # assuming pad_dim as 1
    #Pdb().set_trace()
    pad_dim = 1
    orig_batch_size = target_set.size(0)
    final_batch_size = count.sum()
    rv = []
    for i in range(orig_batch_size):
        this_count = count[i]
        rv.append(target_set[i][:this_count])
    #
    rv = torch.cat(rv)
    assert rv.size(0) == final_batch_size
    return rv


def ravel_and_pad_tensor(target_set, count):
    # TODO: this is padding by 0 and not repeating the last element! couldn't figure out.
    pad_dim = 1
    max_count = count.max()
    cum_sum = count.cumsum(0)
    orig_batch_size = count.size(0)
    rv = []
    for i in range(orig_batch_size):
        start_ind = 0 if i == 0 else cum_sum[i-1]
        end_ind = cum_sum[i]
        rv.append(F.pad(target_set[start_ind:end_ind].transpose(
            0, -1).unsqueeze(0), (0, (max_count - count[i]))).squeeze(0).transpose(0, -1))
    #
    return torch.stack(rv)


def select_feed_dict(feed_dict, select_indices):
    keys = ['mask', 'gtlt', 'n', 'query', 'count', 'is_ambiguous',
            'qid', 'relations', 'target', 'target_set']

    selected_feed_dict = {}
    for key in keys:
        if key in feed_dict:
            selected_feed_dict[key] = feed_dict[key][select_indices]

    return selected_feed_dict


def get_log_prob_from_dis(dis2):
    eps = 0.00001
    return F.log_softmax(-1.0*dis2, dim=0)


def get_prob_from_dis(dis2):
    eps = 0.00001
    if args.latent_model == 'eg':
        return dis2, torch.log(dis2)
    return F.softmax(-1.0*dis2, dim=0), F.log_softmax(-1.0*dis2, dim=0)

def rl_sampling(weights):
    # give weights 1-eps and eps respectively to top2 indices
    #Pdb().set_trace()
    eps = args.exploration_eps
    probs = eps*weights 
    probs[torch.arange(weights.size(0)), weights.argmax(dim=1)] += 1.0 - probs.sum(dim=1)   
    #handling unique case with a hack, assuming that weights[i,0] == 1 whenever ith is unique 
    probs[weights[:,0] == 1] = 0.0
    probs[weights[:,0] == 1,0] = 1.0

    if len(torch.nonzero(torch.abs(probs.sum(dim=1)-1)>eps)):
        print(probs)
        print(weights)
    dist = Categorical(probs)
    return weights.fill_(0.0).scatter_(1,dist.sample().unsqueeze(-1),1.0)

def distributed_iter(selected_feed_dict, start_index, end_index):
    for s, e in zip(start_index, end_index):
        yield_feed_dict = {}
        for k in selected_feed_dict:
            yield_feed_dict[k] = selected_feed_dict[k][s:e]
        yield yield_feed_dict


def update_output(output_dict, this_output_dict):
    if output_dict is None:
        output_dict = this_output_dict
        return output_dict
    for k in this_output_dict:
        if torch.is_tensor(this_output_dict[k]):
            output_dict[k] = torch.cat(
                [output_dict[k], this_output_dict[k]], dim=0)
        else:
            output_dict[k] = output_dict[k].extend(this_output_dict)
    return output_dict


def update_monitors(monitors, this_monitors, count):
    # Pdb().set_trace()
    if monitors is None:
        return this_monitors
    else:
        for k in this_monitors:
            monitors[k] = (count*monitors[k] + this_monitors[k])/(count+1)

    return this_monitors


class MyTrainer(TrainerBase):
    
    def reset_test(self):
        self.pred_dump = []
        self.errors = []
        for i in self.error_distribution:
            self.error_distribution[i]=0
    
    def step(self, feed_dict, reduce_func=default_reduce_func, cast_tensor=False):
        assert self._model.training, 'Step a evaluation-mode model.'
        self.num_iters += 1
        self.trigger_event('step:before', self)
        loss_latent = 0.0
        if cast_tensor:
            feed_dict = as_tensor(feed_dict)

        begin = time.time()

        self.trigger_event('forward:before', self, feed_dict)

        rl_loss = 0.0
        if self.mode == 'warmup':
            loss, monitors, output_dict = self._model(feed_dict)
        else:
            if args.no_static:
                #in no_static scenario, this is for both sampling probabilities and reward.
                #but if latent model is not trainable, we don't need reward and hence setting for_reward to False
                for_reward = self.args.latent_model not in ['det', 'eg']
                loss, monitors, output_dict = self._model(
                    feed_dict, return_loss_matrix=True, for_reward = for_reward)
                y_hat = output_dict['pred'].detach()
            else:
                with torch.no_grad():
                    #y_hat = self._static_model(feed_dict)['pred'].detach()
                    static_model_output = self._static_model(feed_dict, return_loss_matrix=True, for_reward=True)
                    if isinstance(static_model_output, dict):
                        y_hat = static_model_output['pred'].detach()
                        output_dict = static_model_output
                    else:
                        y_hat = static_model_output[2]['pred'].detach()
                        output_dict = static_model_output[2]

            keys = ['mask', 'n', 'query', 'count', 'is_ambiguous',
                    'qid', 'target_set', 'relations', 'gtlt']

            expanded_feed_dict = {}
            for key in keys:
                if key in feed_dict:
                    expanded_feed_dict[key] = expand_tensor(
                        feed_dict[key], feed_dict["count"])
           
            #print("expanded query:\n", expanded_feed_dict['query']) 
            #print("expanded count:\n", expanded_feed_dict['query']) 
            
            #print("original  query:\n", expanded_feed_dict['query']) 
            
            #Pdb().set_trace() 
            #unravel target set to obtain different targets
            expanded_feed_dict["target"] = unravel_tensor(
                feed_dict["target_set"], feed_dict["count"])
            
            
            # copy interemediate y for each target. need not do it only for encoder decoder
            #print("y_hat before expansion: ", y_hat.shape)
            if not (y_hat.shape[0] == feed_dict['count'].sum()): 
                #Pdb().set_trace()
                y_hat = expand_tensor(y_hat, feed_dict["count"])
            #print("y_hat after expansion: ", y_hat.shape)

            # inserting detached loss in the expanded_feed_dict for deterministic latent model
            #Pdb().set_trace()
            if 'loss_matrix' in output_dict:
                expanded_feed_dict['loss'] =  unravel_tensor(output_dict['loss_matrix'], feed_dict['count']).detach()
                if args.latent_model == 'eg':
                    expanded_feed_dict['minloss_eg_prob'] = unravel_minloss_epsilon_greedy(output_dict['loss_matrix'], feed_dict['count'],args.minloss_eg_eps).detach()
            # compute latent variable, i.e. the scores for each of the possible targets
            
            #Pdb().set_trace()
            z_latent = self._latent_model(
                expanded_feed_dict, y_hat,output_dict)['latent_z']

            # start index and end index are markers for start and end indices
            # of each query in the expanded feed dict
            start_index = torch.cumsum(
                feed_dict["count"], 0) - feed_dict["count"]
            end_index = torch.cumsum(feed_dict["count"], 0)

            min_indices = []
            action_prob = []
            #rl_weights = []
            weights = []
            log_softmax = []
            # loop over each query
            for s, e in zip(start_index, end_index):
                dis2 = z_latent[s:e].squeeze(1)
                probs, log_probs = get_prob_from_dis(dis2)
                #print("Dist:\n", dis2)
                #print("Probs:\n", probs)
                weights.append(F.pad(
                    probs, (0, feed_dict['target_set'].size(1) - probs.size(0)), "constant", 0))
                log_softmax.append(F.pad(
                    log_probs, (0, feed_dict['target_set'].size(1) - probs.size(0)), "constant", 0))
            #
            #Pdb().set_trace()
            selected_feed_dict = feed_dict
            if args.rl_exploration:
                selected_feed_dict["weights"] = rl_sampling(torch.stack(weights).detach().clone())
            else:
                selected_feed_dict["weights"] = torch.stack(weights).detach().clone()
                    
            loss = 0
            #Pdb().set_trace()
            for_reward = self.args.latent_model not in ['det', 'eg']
            if (not args.no_static) or (self._model.is_encoder_decoder):
                # Pdb().set_trace() 
                if args.is_encoder_decoder: 
                    if args.no_static: 
                        # M_{theta_} is same as M_{theta} and hence no need to do inference
                        # note that we do not overwrite output_dict that has already been obtained while running inference using M_{theta_}
                        
                        # if for_reward is False, i.e. latent_model is not learnable, then we would have computed the loss while computing the sampling probabilities
                        if for_reward:
                            loss, monitors, _ = self._model(selected_feed_dict)
                        else:
                            #we have already computed loss while computing exploration probabilities
                            loss = (output_dict['loss_matrix']*selected_feed_dict['weights']
                                ).sum()/selected_feed_dict['weights'].sum()

                    else:
                        # here we need to first do inference to compute reward and then forward pass to compute loss
                        raise
                else: 
                    loss, monitors, output_dict = self._model(selected_feed_dict)
            else:
                #Pdb().set_trace()
                loss = (output_dict['loss_matrix']*selected_feed_dict['weights']
                            ).sum()/selected_feed_dict['weights'].sum()

            if (feed_dict['is_ambiguous'].sum() > 0):
                if not args.rl_exploration:
                    avg_reward = ((output_dict['reward']*(feed_dict['mask'].float())).sum(
                    dim=1)/(feed_dict['mask'].sum(dim=1).float())).unsqueeze(-1)
                    #avg_reward = (output_dict['reward']*(feed_dict['mask'].float())).sum()/(feed_dict['mask'].sum().float())
                    rewards = (output_dict['reward'] -
                           avg_reward)*(feed_dict['mask'].float())
                    #print("Rewards:\n", rewards)
                    
                    stacked_exploration_probs = torch.stack(weights)
                    #print("Exploration probs:\n", stacked_exploration_probs)
                    rl_loss = -1.0*(rewards*stacked_exploration_probs).sum()/feed_dict['is_ambiguous'].sum()
                    #rl_loss = -1.0*(rewards*selected_feed_dict['weights']*torch.stack(log_softmax)).sum()/feed_dict['is_ambiguous'].sum()
                else:
                    #use selected_feed_dict['weights']. rewards should be only for non zero samples. 
                    #Also, now we use REINFORCE : maximize : reward*log(p_action)
                    rl_loss = -1.0*((output_dict['reward']+0.5)*selected_feed_dict['weights']*torch.log(torch.stack(weights) + 1.0 - selected_feed_dict['weights'])).sum()/feed_dict['is_ambiguous'].sum().float() 
            loss_latent = rl_loss  

        self.trigger_event('forward:after', self, feed_dict,
                           loss, monitors, output_dict)

        loss = reduce_func('loss', loss)
        loss_f = as_float(loss)

        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}
        if self.mode == 'hot':
            monitors['loss_latent'] = loss_latent
        monitors_f = as_float(monitors)

        self._optimizer.zero_grad()
        if self.mode in ['hot']:
            if torch.is_tensor(loss_latent):
                loss_latent = reduce_func('loss_latent', loss_latent)
            #
            self._latent_optimizer.zero_grad()

        self.trigger_event('backward:before', self,
                           feed_dict, loss, monitors, output_dict)

        if loss.requires_grad:
            loss.backward()

        if self.mode in ['hot']:
            if torch.is_tensor(loss_latent):
                loss_latent.backward()
                # print("Grad:",self._latent_model.digit_embed.weight.grad[2,:2],self._latent_model.atn_across_steps.grad)
                # Pdb().set_trace()
                #print('Latent: ',self.digit_embed.weight.data[2,:4], self.row_embed.weight.data[2,:4])
                #print('Atn over steps: ',self.atn_across_steps)

        self.trigger_event('backward:after', self, feed_dict,
                           loss, monitors, output_dict)

        loss_latent_f = loss_latent.item() if torch.is_tensor(loss_latent) else loss_latent
        grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip, lgrad_norm_before_clip, lgrad_norm_after_clip, lparam_norm_before_clip = 0, 0, 0, -1, -1, 0

        if loss.requires_grad:
            grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip = utils.gradient_normalization(
                self._model, grad_norm=args.grad_clip)
            #glogger.info(','.join(map(lambda x: str(round(x,6)),[self.current_epoch, self.num_iters, loss_f, loss_latent_f, grad_norm_before_clip.item(), grad_norm_after_clip.item(), param_norm_before_clip.item()])))
            if grad_norm_before_clip <= args.upper_limit_on_grad_norm:
                self._optimizer.step()
            else:
                self.num_bad_updates += 1
                logger.info('not taking optim step. Grad too high {}. Num bad updates: {}'.format(round(grad_norm_before_clip,2), self.num_bad_updates))

            #self._optimizer.step()

        if self.mode in ['hot']:
            lgrad_norm_before_clip, lgrad_norm_after_clip, lparam_norm_before_clip = utils.gradient_normalization(
                self._latent_model, grad_norm=args.grad_clip)
            monitors_f['gr_loss_rl'] = lgrad_norm_before_clip 
            self._latent_optimizer.step()

        glogger.info(','.join(map(lambda x: str(round(x, 6)), [self.current_epoch, self.num_iters, loss_f, loss_latent_f, grad_norm_before_clip, grad_norm_after_clip, param_norm_before_clip,lgrad_norm_before_clip, lgrad_norm_after_clip, lparam_norm_before_clip ])))
        end = time.time()

        self.trigger_event('step:after', self)

        return loss_f, monitors_f, output_dict, {'time/gpu': end - begin}

    def save_checkpoint(self, name):

        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir,
                                           'checkpoint_{}.pth'.format(name))

            model = self._model
            if self._latent_model:
                latent_model = self._latent_model
                if not args.no_static:
                    static_model = self._static_model
            else:
                latent_model = None
                if not args.no_static:
                    static_model = None

            if isinstance(model, nn.DataParallel):
                model = model.module
                if latent_model:
                    latent_model = latent_model.module
                    if not args.no_static:
                        static_model = static_model.module

            state = {
                'model': as_cpu(model.state_dict()),
                'optimizer': as_cpu(self._optimizer.state_dict()),
                'extra': {'name': name}
            }

            if latent_model:
                state["latent_model"] = as_cpu(latent_model.state_dict())
                if not args.no_static:
                    state["static_model"] = as_cpu(static_model.state_dict())
                state["latent_optimizer"] = as_cpu(
                    self._latent_optimizer.state_dict())
            try:
                torch.save(state, checkpoint_file)
                logger.info('Checkpoint saved: "{}".'.format(checkpoint_file))
            except Exception:
                logger.exception(
                    'Error occurred when dump checkpoint "{}".'.format(checkpoint_file))

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            model = self._model
            if self._latent_model:
                latent_model = self._latent_model
                if not args.no_static:
                    static_model = self._static_model
            else:
                latent_model = None
                if not args.no_static:
                    static_model = None

            if isinstance(model, nn.DataParallel):
                model = model.module
                if latent_model:
                    latent_model = latent_model.module
                    if not args.no_static:
                        static_model = static_model.module
            try:
                checkpoint = torch.load(filename)
                model.load_state_dict(checkpoint['model'])
                if ("latent_model" in checkpoint) and (latent_model is not None):
                    latent_model.load_state_dict(checkpoint["latent_model"])
                    if not args.no_static:
                        static_model.load_state_dict(
                            checkpoint["static_model"])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception(
                    'Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning(
                'No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['time'] = time.time()
            meters_kv['htime'] = str(datetime.datetime.now())
            meters_kv['config'] = args.dump_dir
            meters_kv['lr'] = self._optimizer.param_groups[0]['lr']
            if mode == 'train':
                meters_kv['epoch'] = self.current_epoch
                meters_kv['data_file'] = args.train_file
            else:
                meters_kv['epoch'] = -1
                meters_kv['data_file'] = args.test_file
                meters_kv['error distribution'] = "-".join(
                    [str(self.error_distribution[k]) for k in sorted(self.error_distribution.keys())])
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    data_iterator = {}
    datasets = {}

    def _prepare_dataset(self, epoch_size, mode):
        assert mode in ['train', 'test']

        if mode == 'train':
            batch_size = args.batch_size
            number = args.train_number
        else:
            batch_size = args.test_batch_size
            number = self.test_number

        # The actual number of instances in an epoch is epoch_size * batch_size.
        #
        if mode in self.datasets:
            dataset = self.datasets[mode]
        else:
            dataset = make_dataset(number, epoch_size *
                                   batch_size, mode == 'train')
            self.datasets[mode] = dataset

        dataloader = JacDataLoader(
            dataset,
            shuffle=(mode == 'train'),
            batch_size=batch_size,
            num_workers=min(epoch_size, 0))
        self.data_iterator[mode] = dataloader.__iter__()

    def ravel_feed_dict(self, feed_dict):
        ret_dict = {}
        pad_count = len(feed_dict["target_set"][0])
        tile_keys = set(feed_dict.keys())
        tile_keys = tile_keys.difference(set(["target_set", "target", "qid"]))

        for key in tile_keys:
            ret_dict[key] = tile(feed_dict[key], 0, pad_count)

        ret_dict["target"] = feed_dict["target_set"].view(
            -1, feed_dict["target_set"].shape[2])
        return ret_dict

    def _get_data(self, index, meters, mode):
        #
        # Pdb().set_trace()
        feed_dict = self.data_iterator[mode].next()

        meters.update(number=feed_dict['n'].data.numpy().mean())
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)
        return feed_dict

    #used in _test
    def _get_result(self, index, meters, mode):
        feed_dict = self._get_data(index, meters, mode)

        # sample latent
        z_latent = None
        # if self.z_list is not None:
        #    z_latent = torch.stack(py_random.sample(self.z_list,feed_dict['query'].size(0))).cuda()
        #    z_latent = z_latent * feed_dict['is_ambiguous'].float().unsqueeze(-1).expand_as(z_latent)

        # provide  ambiguity information about test
        # print(feed_dict['qid'])
        output_dict = self.model(feed_dict, z_latent)
        if not isinstance(output_dict,dict):
            output_dict = output_dict[2]
        if args.test_only:
            self.pred_dump.append(dict(feed_dict=as_cpu(
                feed_dict), output_dict=as_cpu(output_dict)))
        # if mode=="test":
        #    feed_dict["query"] = output_dict["pred"].unsqueeze(-1)
        #    output_dict = self.model(feed_dict)
        target = feed_dict['target']
        result, errors, _ = instance_accuracy(
            target, output_dict['pred'], return_float=True, feed_dict=feed_dict, task=args.task, args=args)
        succ = result['accuracy'] == 1.0

        meters.update(succ=succ)
        meters.update(result, n=target.size(0))
        message = '> {} iter={iter}, accuracy={accuracy:.4f}'.format(
            mode, iter=index, **meters.val)

        if mode == "test":
            self.dump_errors(errors)

        return message, dict(succ=succ, feed_dict=feed_dict)

    def dump_errors(self, errors=None, force=False):
        if errors is not None:
            self.errors.extend(errors)
            for num in errors:
                if num in self.error_distribution:
                    self.error_distribution[num] += 1
                else:
                    self.error_distribution[num] = 1
        if force:
            print("Called with force")
            self.error_distribution = dict(Counter(self.errors))

    def _get_train_data(self, index, meters):
        return self._get_data(index, meters, mode='train')

    def _train_epoch(self, epoch_size, is_last=False):
        meters = super()._train_epoch(epoch_size)

        logger.info("Best Dev Accuracy: {}".format(self.best_accuracy))
        i = self.current_epoch

        if self.mode == "hot" and args.copy_back_frequency > 0 and i % args.copy_back_frequency == 0:

            if not args.no_static:
                logger.info("Copying updated parameters to static model")
                self._static_model = copy.deepcopy(self._model)
                self._static_model.train()

        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        test_meters = None
        if  (is_last or (args.test_interval is not None and i % args.test_interval == 0 and i > args.test_begin_epoch)):
            for i in self.error_distribution:
                self.error_distribution[i]=0
            self.reset_test()
            test_meters = self.test()
            if self.best_accuracy < test_meters[0].avg["corrected accuracy"]:
                self.best_accuracy = test_meters[0].avg["corrected accuracy"]
                if self.checkpoint_mode == "warmup":
                    self.save_checkpoint("best_warmup")
                    self.save_checkpoint("best")
                else:
                    self.save_checkpoint("best")
        return meters, test_meters



    def train(self, start_epoch=1, num_epochs=0):
        meters = None
        for i in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = i
            meters, test_meters = self._train_epoch(
                self.epoch_size, (self.current_epoch == (start_epoch + num_epochs-1)))

            if args.reduce_lr and test_meters is not None:
                #TODO changed to pointwise accuracy for lstm sudoku
                metric = test_meters[0].avg["corrected accuracy"]
                #metric = test_meters[0].avg["pointwise accuracy"]
                self.my_lr_scheduler.step(1.0-1.0*metric)
                if self.my_lr_scheduler.shouldStopTraining():
                    logger.info("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                        self.my_lr_scheduler.unconstrainedBadEpochs, self.my_lr_scheduler.maxPatienceToStopTraining))
                    break

        return meters, test_meters



def test_at_end(trainer):
    logger.info("++++++++ START RUNNING TEST AT END -------")
    test_files = {}
    if args.task_is_sudoku:
        test_files = {'data/sudoku_9_val_e_big_amb.pkl':'val_e_big_amb','data/sudoku_9_val_e_big_unique.pkl':'val_e_big_unq', 'data/sudoku_9_val_d.pkl':'val_d','data/sudoku_9_val_a.pkl': 'val_a','data/sudoku_9_test_e.pkl': 'test_e_all', 'data/sudoku_9_test_e_big_amb.pkl': 'test_e_big_amb', 'data/sudoku_9_test_e_big_unique.pkl': 'test_e_big_unq'}
    #
    if args.task_is_nqueens:
        test_files = {'data/nqueens_11_6_test.pkl': 'test_11_6','data/nqueens_11_6_val.pkl': 'val_11_6'}

    if args.task_is_futoshiki:
        test_files = {'data/futo_6_18_5_test.pkl': 'test_6_18','data/futo_6_18_5_val.pkl': 'val_6_18'}
    args.test_only = 1
    for tf in test_files:
        logger.info("Testing for: {}".format(tf))
        args.test_file = tf
        if 'test' in trainer.datasets:
            del trainer.datasets['test']
        if 'test' in trainer.data_iterator:
            del trainer.data_iterator['test'] 
        trainer.reset_test()
        if args.task_is_nqueens or args.task_is_futoshiki:
            args.test_number = int(test_files[tf].split('_')[-2])
            trainer.test_number_begin = args.test_number
            trainer.test_number_end = args.test_number
        
        rv = trainer.test()
        #with open(os.path.join(args.current_dump_dir, test_files[tf]+"_pred_dump.pkl"), "wb") as f:
        #    pickle.dump(trainer.pred_dump, f)
        with open(os.path.join(args.current_dump_dir, 'results.out'), "a") as f:
            print(tf,test_files[tf],rv[0].avg['corrected accuracy'], file=f)


def main():
    if args.dump_dir is not None:
        args.current_dump_dir = args.dump_dir

        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')
        args.checkpoints_dir = os.path.join(
            args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)

    exp_fh = open(os.path.join(args.current_dump_dir,'exp.sh'),'a')
    print('jac-run {}'.format(' '.join(sys.argv)),file=exp_fh)
    exp_fh.close()

    logger.info('jac-run {}'.format(' '.join(sys.argv))) 
    logger.info(format_args(args))
    print(args.solution_count)
    model = models.get_model(args)

    if args.use_gpu:
        model.cuda()

    if args.get_optim_from_model == 1:
        optimizer = model.get_optimizer(args)
    else:
        optimizer = get_optimizer(args.optimizer, model,
                              args.lr, weight_decay=args.wt_decay)


    trainer = MyTrainer.from_args(model, optimizer, args)
    trainer.args = args
    trainer.num_iters = 0
    trainer.num_bad_updates = 0
    trainer.test_batch_size = args.test_batch_size
    trainer.mode = 'warmup'
    trainer.checkpoint_mode = "warmup"
    trainer._latent_model = None
    trainer._static_model = None



    #skip_warmup = False
    skip_warmup =args.skip_warmup 
    if args.load_checkpoint is not None:
        extra = trainer.load_checkpoint(args.load_checkpoint)
        #skip_warmup = extra is not None and (extra['name'] == 'last_warmup')
        skip_warmup = args.skip_warmup

    #my_lr_scheduler = scheduler.CustomReduceLROnPlateau(trainer._optimizer, {'mode': 'min', 'factor': 0.2, 'patience': math.ceil(
    my_lr_scheduler = scheduler.get_scheduler(args, trainer._optimizer)
    
    trainer.my_lr_scheduler = my_lr_scheduler
    
    if args.test_only:
        #
        # trainer.load_latent_samples(os.path.join(
        # args.current_dump_dir, "latent_z_samples.pkl"))
        trainer.pred_dump = []
        trainer.reset_test()
        rv = trainer.test()
        #with open(os.path.join(args.current_dump_dir, "pred_dump.pkl"), "wb") as f:
        #    pickle.dump(trainer.pred_dump, f)
        trainer.dump_errors(force=True)
        with open(os.path.join(args.current_dump_dir, 'results.out'), "w") as f:
            print(rv[0].avg['corrected accuracy'], file=f)

        test_at_end(trainer)
        return None, rv

    if not skip_warmup:
        warmup_meters, warmup_test_meters = trainer.train(
            1, args.warmup_epochs)
        trainer.save_checkpoint('last_warmup')
    else:
        logger.info("Skipping warmup")

    if args.epochs > 0:
        # define latent model
        # clone the main model
        # set the optimizer
        if skip_warmup:
            trainer._prepare_dataset(args.epoch_size, 'train')
        #
        trainer.checkpoint_mode = "hot"
        trainer.best_accuracy = -1
        args.min_loss = 0

        trainer._latent_model = models.get_latent_model(
            args, trainer.model)
        trainer._latent_model.train()
        if not args.no_static:
            trainer._static_model = copy.deepcopy(trainer._model)
        
        trainer._latent_optimizer = get_optimizer(
            args.optimizer, trainer._latent_model, args.lr_latent, weight_decay=args.latent_wt_decay)

        trainer.mode = "hot"

        # switch off training mode only after pretraining phi
        # since pretraining phi requires training statistics
        if not args.no_static:
            trainer._static_model.eval()
            #trainer._static_model.training = True
        #
        # if skip_warmup:
        #    extra = trainer.load_checkpoint(args.load_checkpoint)
        trainer.datasets['train'].reset_sampler(args.hot_data_sampling)
        #trainer.datasets["train"].data_sampling = args.hot_data_sampling
   
        if not args.no_static:
            trainer._static_model.train()
        if args.pretrain_phi > 0:
            my_lr_scheduler.maxPatienceToStopTraining = 10000
            for x in trainer._optimizer.param_groups:
                x['lr'] = 0.0
            print("Start pretraining of phi")
            _ = trainer.train(args.warmup_epochs+1, args.pretrain_phi)
            print("End pretraining of phi")
        
        trainer.best_accuracy = -1

        trainer._optimizer = get_optimizer(
            args.optimizer, trainer.model, args.lr_hot, weight_decay=args.wt_decay)

        my_lr_scheduler = scheduler.CustomReduceLROnPlateau(trainer._optimizer, {'mode': 'min', 'factor': 0.2, 'patience': math.ceil(
            7/args.test_interval), 'verbose': True, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0.01*args.lr_hot, 'eps': 0.0000001}, maxPatienceToStopTraining=math.ceil(25/args.test_interval))
        trainer.my_lr_scheduler = my_lr_scheduler

        final_meters = trainer.train(
            args.warmup_epochs+args.pretrain_phi+1, args.epochs)
        trainer.save_checkpoint('last')

    trainer.load_checkpoint(os.path.join(
        args.checkpoints_dir, 'checkpoint_best.pth'))
    logger.info("Best Dev Accuracy: {}".format(trainer.best_accuracy))

   
    trainer.reset_test()
    ret = trainer.test()
    trainer.dump_errors(force=True)
    with open(os.path.join(args.current_dump_dir, 'results.out'), "w") as f:
        print(trainer.best_accuracy, ret[0].avg['corrected accuracy'], file=f)

    test_at_end(trainer)
    return ret


if __name__ == '__main__':
    _ = main()
