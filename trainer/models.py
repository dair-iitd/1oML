import math
import pickle
import copy
import collections
import functools
import os
import json
import time
import datetime
import dgl
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
from difflogic.nn.neural_logic import LogicMachine, LogicInference, LogitsInference, LogicSoftmaxInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.thutils_rl import binary_accuracy, instance_accuracy
from difflogic.train import TrainerBase

from rrn.sudoku import SudokuNN
import rrn.sudoku_data as sd
from lstm_encoder_decoder_sudoku import LSTMSudokuSolver
from transformer_encoder_decoder_sudoku import TransformerSudokuSolver
from t5_encoder_decoder_sudoku import MyT5SudokuSolver  
from t5v2_encoder_decoder_sudoku import MyT5V2SudokuSolver  
from t5v3_encoder_decoder_sudoku import MyT5V3SudokuSolver  
from t5_encoder_sudoku import MyT5EncoderSudokuSolver 
from gpt_encoder_sudoku import MyGPTEncoderSudokuSolver 
from gpt_encoder_decoder_sudoku import MyGPTEncoderDecoderSudokuSolver 
from latent_models import SudokuConvNet, EpsilonGreedyLatentModel, DeterministicLatentModel, LatentNLMModel
import utils
from IPython.core.debugger import Pdb 
logger = get_logger(__file__)


def get_model(args):
    if args.model == 'nlm':
        rmodel = NLMModel(args)
    elif args.model == 'rrn':
        rmodel = SudokuRRNNet(args)
    elif args.model in ['transformer_encoder_decoder_sudoku', 'lstm_encoder_decoder_sudoku', 
            't5_encoder_decoder_sudoku',
            't5v2_encoder_decoder_sudoku',
            't5v3_encoder_decoder_sudoku',
            't5_encoder_sudoku',
            'gpt_encoder_sudoku',
            'gpt_encoder_decoder_sudoku']:
        rmodel = SudokuEncoderDecoder(args)
    if args.use_gpu:
        rmodel = rmodel.cuda()
    return rmodel


def get_latent_model(args, base_model=None):
    if args.latent_model == 'eg':
        rmodel = EpsilonGreedyLatentModel(args)
    elif args.latent_model == 'det':
        rmodel = DeterministicLatentModel(args)
    elif args.model == 'nlm':
        rmodel = LatentNLMModel(args)
    elif args.latent_model == 'conv':
        rmodel = SudokuConvNet(args)
    if args.use_gpu:
        rmodel = rmodel.cuda()
    return rmodel


class NLMModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.is_encoder_decoder=False

        # inputs
        self.args = args
        if args.task_is_nqueens:
            binary_dim = 4
            unary_dim = 1
        elif args.task_is_futoshiki:
            binary_dim = 3
            unary_dim = 3

        self.feature_axis = 1 if (args.task_is_1d_output) else 2
        input_dims = [0 for _ in range(args.nlm_breadth + 1)]
        input_dims[0] = 0
        input_dims[1] = unary_dim
        input_dims[2] = binary_dim
        self.features = LogicMachine.from_args(
            input_dims, args.nlm_attributes, args, prefix='nlm')

        output_dim = self.features.output_dims[self.feature_axis]
        target_dim = 1
        self.pred = LogicInference(output_dim, target_dim, [])
        #Pdb().set_trace()
        # losses
        self.base_loss = nn.BCELoss()
        self.wt_base_loss = nn.BCELoss(reduction='none')

        def loss_aggregator(pred, target, count, weights=None):
            # if pred and target have same dimension then simply compute loss
            # Pdb().set_trace()
            if pred.dim() == target.dim():
                # if weights are not none then weigh each datapoint appropriately else simply average them
                if weights is not None:
                    loss = (weights*self.wt_base_loss(pred,
                                                      target).sum(dim=1)).sum()/weights.sum()
                    return loss
                return self.base_loss(pred, target)

            if self.args.cc_loss:
                #loss = log(sum_prob)
                # first compute probability of each of the targets
                # pred.shape = BS X N
                # target.shape = BS x num targets x N
                target_prob = pred.unsqueeze(1).expand_as(target)
                target_prob = (target_prob*target.float() + (1 -
                                                             target_prob) * (1 - target.float())).prod(dim=-1)
                # now target_prob is of shape: batchsize x number of targets.
                batch_loss = []
                for i in range(len(target_prob)):
                    total_prob = target_prob[i][:count[i]].sum()
                    batch_loss.append(-1*torch.log(total_prob)/target.size(-1))
                return torch.stack(batch_loss)
            #
            else:
                # if pred and target are not of same dimension then compute loss wrt each element in target set
                # return a (batchsize x targetset size) vector
                batch_loss = []
                for i in range(len(pred)):
                    x = pred[i]
                    instance_loss = []
                    for y in target[i][:count[i]]:
                        instance_loss.append(self.base_loss(x, y))
                    if self.args.min_loss:
                        batch_loss.append(
                            torch.min(torch.stack(instance_loss)))
                    elif self.args.naive_pll_loss:
                        batch_loss.append(torch.mean(
                            torch.stack(instance_loss)))
                    else:
                        batch_loss.append(
                            F.pad(torch.stack(instance_loss), (0, len(target[i])-count[i]), "constant", 0))
                return torch.stack(batch_loss)

        self.loss = loss_aggregator

    def distributed_pred(self, inp, depth):
        feature = self.features(inp, depth=depth)[self.feature_axis]
        pred = self.pred(feature)
        pred = pred.squeeze(-1)
        return pred

    def forward(self, feed_dict, return_loss_matrix=False, for_reward=False):
        feed_dict = GView(feed_dict)

        states = None

        # relations
        relations = feed_dict.relations.float()
        states = feed_dict.query.float()
        batch_size, nr = relations.size()[:2]
        inp = [None for _ in range(self.args.nlm_breadth + 1)]

        inp[1] = states
        inp[2] = relations
        depth = None
        if self.args.nlm_recursion:
            depth = 1
            while 2**depth + 1 < nr:
                depth += 1
            depth = depth * 2 + 1

        pred = self.distributed_pred(inp, depth=depth)

        if self.training:
            monitors = dict()
            target = feed_dict.target
            target = target.float()
            count = None
            if self.args.cc_loss or self.args.min_loss or self.args.naive_pll_loss or 'weights' in feed_dict or return_loss_matrix:
                target = feed_dict.target_set
                target = target.float()
                count = feed_dict.count.int()

            this_meters, _, reward = instance_accuracy(
                feed_dict.target.float(), pred, return_float=False, 
                feed_dict=feed_dict, task=self.args.task, args=self.args)

            #logger.info("Reward: ")
            # logger.info(reward)
            monitors.update(this_meters)
            loss_matrix = self.loss(pred, target, count)

            if self.args.min_loss or self.args.cc_loss or self.args.naive_pll_loss:
                loss = loss_matrix.mean()
            elif 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum() / \
                    feed_dict.weights.sum()
            else:
                loss = loss_matrix

            return loss, monitors, dict(pred=pred, reward=reward, loss_matrix=loss_matrix)
        else:
            return dict(pred=pred)


class SudokuRRNNet(nn.Module):
    def __init__(self, args):
        super(SudokuRRNNet, self).__init__()
        self.is_encoder_decoder = False

        self.args = args
        self.num_steps = args.sudoku_num_steps
        self.sudoku_solver = SudokuNN(
            num_steps=args.sudoku_num_steps,
            embed_size=args.sudoku_embed_size,
            hidden_dim=args.sudoku_hidden_dim,
            edge_drop=args.sudoku_do
        )

        self.basic_graph = sd._basic_sudoku_graph()
        self.sudoku_indices = torch.arange(0, 81)
        if args.use_gpu:
            self.sudoku_indices = self.sudoku_indices.cuda()
        self.rows = self.sudoku_indices // 9
        self.cols = self.sudoku_indices % 9
        self.wt_base_loss = torch.nn.CrossEntropyLoss(reduction='none')

        def loss_aggregator(pred, target, target_mask):
            # if pred and target have same dimension then simply compute loss
            # pred.shape == BS x 10 x 81 x  32
            # target.shape = BS X Target size x 81
            # if pred.dim()==self.args.sudoku_num_steps*target.dim():
            #    return self.base_loss(pred, target)
            # Pdb().set_trace()
            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector
            # target.shape = batch_size x target size x num_variables
            # pred.shape: batch_size x num variables
            # Pdb().set_trace()
            batch_size, target_size, num_variables = target.size()
            num_steps = pred.size(-1)
            #target= torch.stack([target.transpose(1,2)]*num_steps,dim=-1).transpose(-1,-2)
            if self.args.cc_loss:
                # Pdb().set_trace()
                log_pred_prob = F.log_softmax(pred, dim=1)
                epsilon = 1e-10
                #pred_prob = pred_prob.unsqueeze(-1).expand(*pred_size(),target_size)
                target = target.unsqueeze(-1).expand(*target.size(), num_steps)
                #target.shape = bs x noof_targets x 81 x num_steps

                # Pdb().set_trace()
                log_target_prob = torch.gather(
                    log_pred_prob, dim=1, index=target.long()).sum(dim=-2)
                #log_target_prob.shape = bs x noof_targets x num_steps

                # multiply the probability accross cells. dim = -2.
                # target_prob.shape  = BS x Target Size x num cells x num steps
                #target_prob = target_prob.prod(dim=-2)
                #log_target_prob = torch.log(target_prob).sum(dim=-2)
                expanded_mask = target_mask.float().unsqueeze(-1).expand_as(log_target_prob)
                #mask_log_target_prob = log_target_prob*(target_mask.float().unsqueeze(-1).expand_as(log_target_prob))
                # add probabilities for all targets
                #target_prob = expanded_mask*(epsilon + torch.exp(log_target_prob))
                # Pdb().set_trace()
                log_max_prob, max_prob_index = log_target_prob.max(dim=1)
                #log_max_prob =  log_target_prob.gather(dim=1,index=max_prob_index.unsqueeze(1)).squeeze(1)
                # log(sum) = log(sum*max/max) = log(max) + log(sum/max) = log(max) + log(sum(exp(log(p_i) - log(p_max)))

                log_total_prob = log_max_prob + \
                    torch.log((expanded_mask*torch.exp(log_target_prob -
                                                       log_max_prob.unsqueeze(dim=1))).sum(dim=1))

                #log_total_prob = log_max_prob + torch.log((target_prob.sum(dim=1))/max_prob)
                # log_total_prob = log_target_prob[:,0,:] + torch.log(
                #                1.0 + (target_prob[:,1:,:].sum(dim=1)/target_prob[:,0,:]))
                loss_tensor = (-1.0*log_total_prob).mean(dim=-1)/num_variables
                # total_target_prob.shape = BS x Num steps
                #loss_tensor = (-1.0*torch.log(total_target_prob + epsilon )).mean(dim=-1)/num_variables
            else:
                pred = pred.unsqueeze(-1).expand(*pred.size(), target_size)
                target = target.transpose(1, 2).unsqueeze(-1).expand(
                    batch_size, num_variables, target_size, num_steps).transpose(-1, -2)
                loss_tensor = self.wt_base_loss(pred, target.long())
                loss_tensor = loss_tensor.mean(
                    dim=list(range(1, loss_tensor.dim()-1)))*target_mask.float()

                # shape = batch_size x target_size
                if self.args.min_loss:
                    # return has shape: batch_size
                    #loss_tensor  = loss_tensor.masked_fill((1-target_mask.byte()),float('inf')).min(dim=1)[0]
                    loss_tensor = loss_tensor.masked_fill(
                        (target_mask < 1), float('inf')).min(dim=1)[0]
                elif self.args.naive_pll_loss:
                    loss_tensor = loss_tensor.sum(
                        dim=1)/target_mask.sum(dim=1).float()

            return loss_tensor
        self.loss_func = nn.CrossEntropyLoss()
        self.loss = loss_aggregator

    def collate_fn(self, feed_dict):
        graph_list = []
        for i in range(len(feed_dict['query'])):
            # @TODO may have to change dtype of q. keep an eye
            q = feed_dict['query'][i]
            graph = copy.deepcopy(self.basic_graph)
            graph.ndata['q'] = q  # q means question
            #graph.ndata['a'] = feed_dict['target'][i].long()
            graph.ndata['row'] = self.rows
            graph.ndata['col'] = self.cols
            graph_list.append(graph)
        batch_graph = dgl.batch(graph_list)
        return batch_graph

    def forward(self, feed_dict, return_loss_matrix=False,  for_reward=False):
        # Pdb().set_trace()
        feed_dict = GView(feed_dict)
        # convert it to graph
        bg = self.collate_fn(feed_dict)
        # logits : of shape : args.sudoku_num_steps x batchsize*81 x 10 if training
        # logits: of shape : batch_size*81 x 10 if not training
        logits = self.sudoku_solver(bg, self.training)

        if self.training:
            # testing
            """
            labelsa = bg.ndata['a']
            labelsb = torch.stack([labelsa]*self.num_steps, 0)
            labels = labelsb.view([-1])
            labels1 = feed_dict.target.flatten().unsqueeze(0).expand(self.num_steps,-1).flatten().long()
            gl = dgl.unbatch(bg)
            gl[0].ndata['q']
            gl[1].ndata['q']
            Pdb().set_trace()
            print((labels != labels1).sum())
            loss = self.loss_func(logits.view([-1,10]), labels)
            #
            """
            logits = logits.transpose(1, 2)
            logits = logits.transpose(0, 2)
        else:
            logits = logits.unsqueeze(-1)
        # shape of logits now : BS*81 x 10 x 32 if self.training ,  otherwise BS*81 x 10 x 1
        logits = logits.view(-1, 81, logits.size(-2), logits.size(-1))
        # shape of logits now : BS x  81 x 10 x 32(1)
        logits = logits.transpose(1, 2)
        # shape of logits now : BS x  10 x 81 x 32(1)
        #pred = logits[:,:,:,-1].argmax(dim=1)
        pred = logits

        if self.training:
            # Pdb().set_trace()
            this_meters, _, reward = instance_accuracy(feed_dict.target.float(
            ), pred, return_float=False, feed_dict=feed_dict, task=self.args.task, args=self.args)

            monitors = dict()
            target = feed_dict.target.float()
            count = None
            # Pdb().set_trace()
            loss_matrix = None
            if self.args.cc_loss or self.args.naive_pll_loss or self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                loss_matrix = self.loss(
                    logits, feed_dict.target_set, feed_dict.mask)
            else:
                loss_matrix = self.loss(logits, target.unsqueeze(
                    1), feed_dict.mask[:, 0].unsqueeze(-1))
            # Pdb().set_trace()
            if 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum() / \
                    feed_dict.weights.sum()

            else:
                loss = loss_matrix.mean()

            #loss = loss_ch
            # print(loss,loss_ch)

            #logger.info("Reward: ")
            # logger.info(reward)
            monitors.update(this_meters)
            # logits = logits.view([, 10])
            #labels = labels.view([-1])

            # loss_matrix of size: batch_size x target set size
            # when in training mode return prediction for all steps
            return loss, monitors, dict(pred=pred, reward=reward, loss_matrix=loss_matrix)
        else:
            return dict(pred=pred)


class SudokuEncoderDecoder(nn.Module):
    def __init__(self, args):
        super(SudokuEncoderDecoder, self).__init__()
        self.args = args
        self.num_decoder_layers = args.sudoku_num_steps
        self.is_encoder_decoder = True
        if args.model == 'lstm_encoder_decoder_sudoku':
            self.sudoku_solver = LSTMSudokuSolver(
                num_decoder_layers=args.sudoku_num_steps,
                num_encoder_layers = args.sudoku_num_steps,
            )
        elif args.model == 'transformer_encoder_decoder_sudoku':
            self.sudoku_solver = TransformerSudokuSolver(
                num_decoder_layers=args.sudoku_num_steps,
                num_encoder_layers = args.sudoku_num_steps,
            )
        elif args.model == 't5_encoder_decoder_sudoku':
            self.sudoku_solver = MyT5SudokuSolver(args)
        elif args.model == 't5v2_encoder_decoder_sudoku':
            self.sudoku_solver = MyT5V2SudokuSolver(args)
        elif args.model == 't5v3_encoder_decoder_sudoku':
            self.sudoku_solver = MyT5V3SudokuSolver(args)
        elif args.model == 't5_encoder_sudoku':
            self.is_encoder_decoder = False 
            self.sudoku_solver = MyT5EncoderSudokuSolver(args)
        elif args.model == 'gpt_encoder_sudoku':
            self.is_encoder_decoder = False 
            self.sudoku_solver =  MyGPTEncoderSudokuSolver(args)
        elif args.model == 'gpt_encoder_decoder_sudoku':
            self.sudoku_solver =  MyGPTEncoderDecoderSudokuSolver(args)

        self.args.is_encoder_decoder = self.is_encoder_decoder
        self.wt_base_loss = torch.nn.CrossEntropyLoss(reduction='none')

        def loss_aggregator(pred, target, target_mask):
            # if pred and target have same dimension then simply compute loss
            # pred.shape == BS x 10 x 81 x  32
            # target.shape = BS X Target size x 81
            # if pred.dim()==self.args.sudoku_num_steps*target.dim():
            #    return self.base_loss(pred, target)
            # Pdb().set_trace()
            # if pred and target are not of same dimension then compute loss wrt each element in target set
            # return a (batchsize x targetset size) vector
            # target.shape = batch_size x target size x num_variables
            # pred.shape: batch_size x num variables
            # Pdb().set_trace()
            
            batch_size, target_size, num_variables = target.size()
            #shape of target: BS x noof_targets x 81
            # shape of pred : BS x  10 x 81 x 32(1) x noof_targets
            num_steps = pred.size(-2)
            #target= torch.stack([target.transpose(1,2)]*num_steps,dim=-1).transpose(-1,-2)
            if self.args.cc_loss:
                # Pdb().set_trace()
                log_pred_prob = F.log_softmax(pred, dim=1)
                epsilon = 1e-10
                #pred_prob = pred_prob.unsqueeze(-1).expand(*pred_size(),target_size)
                target = target.unsqueeze(-1).expand(*target.size(), num_steps)
                #target.shape == BS x noof_targets x 81 x num_steps (32 or 1)
                #log_pred_prob.shape : BS x 10 x 81 x num_steps 32(1) x noof_targets
                #Pdb().set_trace()
                log_target_prob = torch.gather(
                    log_pred_prob, dim=1, index=target.unsqueeze(-1).transpose(1,-1).long()).sum(dim=2)
                #index is of dimension: BS X 1 X 81 X NUM_STEPS X NOOF_TARGETS
                # multiply the probability accross cells. dim = 2 with 81 size.
                #log_target_prob.shape = bs x 1 x num_steps x noof_targets.
                #Pdb().set_trace() 
                log_target_prob = log_target_prob.squeeze(dim=1)
                #log_target_prob.shape = bs x num_steps x noof_targets.

                log_target_prob = log_target_prob.transpose(1,2)
                #log_target_prob.shape = bs x noof_targets x num_steps

                expanded_mask = target_mask.float().unsqueeze(-1).expand_as(log_target_prob)
                #expanded_mask.shape = bs x noof_targets x num_steps

                log_max_prob, max_prob_index = log_target_prob.max(dim=1)
                #max at each step separately
                #log_max_prob.shape = bs x num_steps

                # log(sum) = log(sum*max/max) = log(max) + log(sum/max) = log(max) + log(sum(exp(log(p_i) - log(p_max)))

                #print("Max log prob at the last step: " , log_max_prob[:,-1])
                
                log_total_prob = log_max_prob + \
                    torch.log((expanded_mask*torch.exp(log_target_prob -
                                                       log_max_prob.unsqueeze(dim=1))).sum(dim=1))

                #print("Total log prob at the last step: " , log_total_prob[:,-1])

                
                #log_total_prob = log_max_prob + torch.log((target_prob.sum(dim=1))/max_prob)
                # log_total_prob = log_target_prob[:,0,:] + torch.log(
                #                1.0 + (target_prob[:,1:,:].sum(dim=1)/target_prob[:,0,:]))
                loss_tensor = (-1.0*log_total_prob).mean(dim=-1)/num_variables
                # total_target_prob.shape = BS x Num steps
                #loss_tensor = (-1.0*torch.log(total_target_prob + epsilon )).mean(dim=-1)/num_variables
            else:
               
                #for each target, preds/logits are now different, so we don't need to expand.. it's last dimension should match target_size
                #pred = pred.unsqueeze(-1).expand(*pred.size(), target_size)
                target = target.transpose(1, 2).unsqueeze(-1).expand(
                    batch_size, num_variables, target_size, num_steps).transpose(-1, -2)
                loss_tensor = self.wt_base_loss(pred, target.long())
                loss_tensor = loss_tensor.mean(
                    dim=list(range(1, loss_tensor.dim()-1)))*target_mask.float()

                # shape = batch_size x target_size
                if self.args.min_loss:
                    # return has shape: batch_size
                    #loss_tensor  = loss_tensor.masked_fill((1-target_mask.byte()),float('inf')).min(dim=1)[0]
                    loss_tensor = loss_tensor.masked_fill(
                        (target_mask < 1), float('inf')).min(dim=1)[0]
                elif self.args.naive_pll_loss:
                    loss_tensor = loss_tensor.sum(
                        dim=1)/target_mask.sum(dim=1).float()

            return loss_tensor
        self.loss_func = nn.CrossEntropyLoss()
        self.loss = loss_aggregator

    def collate_fn(self, feed_dict, return_loss_matrix, for_reward):
        inputs = feed_dict['query'].reshape(-1,9,9)

        if self.training:
            if (not for_reward) and (self.is_encoder_decoder) and (self.args.cc_loss or self.args.naive_pll_loss or self.args.min_loss or 'weights' in feed_dict or return_loss_matrix):
                #raise
                #here we need to create pairs of inp, out based on mask
                inputs_list = []
                outputs_list = []
                num_samples, num_targets = feed_dict['mask'].shape
                sample_no_list = []
                for i in range(num_samples):
                    for j in range(num_targets):
                        if feed_dict['mask'][i,j] != 0:
                            inputs_list.append(feed_dict['query'][i])
                            outputs_list.append(feed_dict['target_set'][i,j])
                            sample_no_list.append(i) 
                #
                inputs = torch.stack(inputs_list)
                outputs = torch.stack(outputs_list)
                inputs = inputs.reshape(-1,9,9)
                outputs = outputs.reshape(-1,9,9)
                return inputs.to(feed_dict['query'].device).long(), outputs.to(feed_dict['query'].device).long(), sample_no_list
            else:
                #sample_no_list = [i for i in range(feed_dict['query'].shape[0]] 
                outputs = feed_dict['target'].reshape(-1,9,9)
                return inputs.long(), outputs.long(), None 

        else:
            return inputs.long(), None, None

    def get_optimizer(self, args):
        return self.sudoku_solver.get_optimizer(args)
        
    def forward(self, feed_dict, return_loss_matrix=False, for_reward=False):
        # Pdb().set_trace()
        feed_dict = GView(feed_dict)
        # convert it to batch
        #inputs.shape: batch_size x 9 x 9
        #outputs.shape: batch_size x 9 x 9
        inputs, outputs, sample_no_list = self.collate_fn(feed_dict, return_loss_matrix, for_reward)

        logits = self.sudoku_solver(inputs, outputs, self.training, for_reward)
        # shape of logits now : BS x  10 x 81 x 32(1)
        
        pred = logits
        
        # earlier, logits would be same for a given x, irrespective of the target y. 
        # But in autoregressive model, logits are different for each (x,y) pair.
        # Hence, we need to modify our loss computation which earlier projected the same logits..

        if self.training:
            #Pdb().set_trace()
            #this_meters, _, reward = instance_accuracy(feed_dict.target.float(
            #), pred, return_float=False, feed_dict=feed_dict, task=self.args.task, args=self.args)
             
            this_meters, _, reward = instance_accuracy(outputs.float(
            ).view(outputs.shape[0],-1), pred, return_float=False, feed_dict=feed_dict, task=self.args.task, args=self.args, inputs=inputs.view(inputs.shape[0],-1))

            monitors = dict()
            target = feed_dict.target.float()
            count = None
            #Pdb().set_trace()
            loss_matrix = None
            if self.args.cc_loss or self.args.naive_pll_loss or self.args.min_loss or 'weights' in feed_dict or return_loss_matrix:
                #raise
                if self.is_encoder_decoder and (not for_reward):
                    #need to create logits tensor. Use sample_no_list for this
                    total_no_of_samples = 1+sample_no_list[-1]
                    pred_list = [[] for _ in range(total_no_of_samples)]
                    for ind, sample_no in enumerate(sample_no_list):
                        pred_list[sample_no].append(logits[ind])
                    #
                    max_no_of_targets = feed_dict.mask.shape[-1]
                    for this_sample in range(total_no_of_samples):
                        to_add  = max_no_of_targets - len(pred_list[this_sample])
                        for _ in range(to_add):
                            pred_list[this_sample].append(pred_list[this_sample][-1])
                        pred_list[this_sample] = torch.stack(pred_list[this_sample], dim=-1)
                    #
                    stacked_logits = torch.stack(pred_list, dim=0)
                    #Pdb().set_trace()
                else:
                    stacked_logits = logits.unsqueeze(-1).expand(*logits.size(), feed_dict.mask.shape[1])
                loss_matrix = self.loss(
                    stacked_logits, feed_dict.target_set, feed_dict.mask)
            else:
                logits = logits.unsqueeze(-1)
                loss_matrix = self.loss(logits, target.unsqueeze(
                    1), feed_dict.mask[:, 0].unsqueeze(-1))
            # Pdb().set_trace()
            if 'weights' in feed_dict:
                loss = (feed_dict.weights*loss_matrix).sum() / \
                    feed_dict.weights.sum()
            else:
                loss = loss_matrix.mean()

            #loss = loss_ch
            # print(loss,loss_ch)

            #logger.info("Reward: ")
            # logger.info(reward)
            monitors.update(this_meters)
            # logits = logits.view([, 10])
            #labels = labels.view([-1])

            # loss_matrix of size: batch_size x target set size
            # when in training mode return prediction for all steps
            return loss, monitors, dict(pred=pred, reward=reward, loss_matrix=loss_matrix)
        else:
            return dict(pred=pred)
